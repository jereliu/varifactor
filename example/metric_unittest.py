import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pickle as pk

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as st
from scipy.stats import norm
import theano

from varifactor.model import NEFactorModel as Model
from varifactor.metric.kernel import KSD
from varifactor.util.kernel import RBF
from varifactor.util.result_handler import read_npy, get_eigen
from varifactor.util.setting import param_model

from varifactor.util.decorator import add_plot_option

####################
# 1. read in file  #
####################

res_addr = "./result/Poisson_n50_p5_k2/"
report_addr = \
    "/home/jeremiah/Dropbox/Research/Harvard/Thesis/Lorenzo/" \
    "1. varifactor/Report/Progress/2018_03_Week_2/plot/"

method_list = ['Metropolis', 'NUTS', 'ADVI']
# method_list = ['Metropolis', 'NUTS', 'ADVI', 'NFVI']
# method_list = os.listdir(res_addr)

U = dict()
V = dict()
Y = dict()
F = dict()
eig_F = dict()
T = dict()

for method in method_list:
    res_addr = "./result/Poisson_n50_p5_k2/%s/" % (method)
    print("%s: U" % method)
    U[method] = read_npy(res_addr + "U/")
    print("%s: V" % method)
    V[method] = read_npy(res_addr + "V/")
    print("%s: F" % method)
    F[method] = np.concatenate((U[method], V[method]), axis=-2)
    eig_F[method] = get_eigen(F[method])
    print("%s: Y" % method)
    Y[method] = read_npy(res_addr + "Y/")
    print("%s: time" % method)
    T[method] = np.mean(read_npy(res_addr + "time/"))

pk.dump(eig_F, open("./result/eig_F.pkl", "wr"))

#####################################
# 2. helper functions               #
#####################################


def moment_distance(sample, type="mean", sd_true=1):
    if type == "llik":
        # likelihood samples with 2 dim
        dist_iter = np.mean(sample, axis = 0)
    else:
        # otherwise raw samples with 4 dim
        n_sample, n_iter, dim1, dim2 = sample.shape

        if type == "mean":
            moment_true = np.zeros(shape=(dim1, dim2))
            moment_sample = np.mean(sample, axis=0)

            dist_iter = np.sqrt(
                np.mean((moment_sample - moment_true) ** 2, axis=(-2, -1)))
        elif type == "cov":
            moment_true = np.eye(dim2) * sd_true**2
            sample_temp = np.swapaxes(sample, 0, 1).reshape(n_iter, n_sample*dim1, dim2)
            moment_sample = np.array([np.cov(sample_temp[i].T) for i in range(n_iter)])

            dist_iter = np.sqrt(
                np.mean((moment_sample - moment_true) ** 2, axis=(-2, -1)))

        elif type == "prob":
            dist_iter = np.zeros(shape=n_iter)
            for iter_id in range(n_iter):
                sample_iter = sample[:, iter_id, :, :]
                moment_sample_upper = np.percentile(sample_iter, 97.5, axis=0)
                moment_sample_lower = np.percentile(sample_iter, 2.5, axis=0)
                moment_sample = \
                    norm.cdf(moment_sample_upper, loc=0, scale=sd_true) - \
                    norm.cdf(moment_sample_lower, loc=0, scale=sd_true)
                dist_iter[iter_id] = np.mean(moment_sample)

        elif type == "llik_prior":
            dist_iter = np.zeros(shape=n_iter)
            for iter_id in range(n_iter):
                sample_iter = sample[:, iter_id, :, :]
                moment_sample = np.mean(norm.logpdf(sample_iter))
                dist_iter[iter_id] = moment_sample

    return dist_iter


def moment_measure(sample, type="mean", n_boot=500, sd_true=1):
    """
    compute empirical measures as well as bootstrap percentiles (5%, 50%, 95%)
    :param sample:
    :param type:
    :param n_boot:
    :param sd_true:
    :return:
    """
    n_chain, n_iter = sample.shape[:2]

    # compute mean
    measure_mean = moment_distance(sample, type=type, sd_true=sd_true)

    # compute bootstrap sample
    measure_boot_sample = np.zeros(shape=(n_boot, n_iter))
    for boot_id in tqdm(range(n_boot)):
        boot_chain_id = np.random.choice(n_chain, n_chain)
        measure_boot_sample[boot_id] = \
            moment_distance(sample[boot_chain_id], type=type, sd_true=sd_true)

    measure_pc = np.percentile(measure_boot_sample, (5, 50, 95), axis=0).T
    measure_all = \
        np.concatenate((np.atleast_2d(measure_mean).T, measure_pc), axis=1)

    return measure_all


def compute_llik(u, v, y, lik_fun):
    """

    :param u: n_chain x n_iter x N x K tensor of latent traits
    :param v: n_chain x n_iter x P x K tensor of factor loading
    :param y: n_chain x N x P tensor of observation
    :param lik_fun: likelihood function
    :return:
    """
    n_chain, n_iter, _, _ = u.shape
    lik_list = np.zeros(shape=(n_chain, n_iter))

    for chain_id in tqdm(range(n_chain)):
        y_chain = y[chain_id]
        for iter_id in range(n_iter):
            u_iter = u[chain_id, iter_id]
            v_iter = v[chain_id, iter_id]
            args = {'u': u_iter, 'v': v_iter, 'y': y_chain}
            lik_list[chain_id, iter_id] = lik_fun(**args)

    return lik_list


def iter_smooth(sample, span=5, method="mean"):
    """
    helper function to smooth sample measurements over iterations

    :param sample: np.array with dimension n_iter x n_stat
    :param span: number of iteration to include for local averaging
    :param method: smoothing method to be used
    :return:
    """
    # prepare meta variables
    span_radius = int((span-1)/2)
    if span_radius == 0:
        return sample

    if method == "mean":
        smooth_fun = np.mean
    elif method == "median":
        smooth_fun = np.median
    else:
        raise ValueError("method %s not supported" % method)

    n_iter, n_stat = sample.shape

    sample_smooth = np.zeros(shape=(n_iter, n_stat))
    for stat_id in range(n_stat):
        stat = sample[:, stat_id]
        stat_smooth = stat.copy()
        for iter_id in range(1, n_iter):
            iter_lw = np.max((iter_id - span_radius, 1))
            iter_up = np.min((iter_id + span_radius, n_iter))
            stat_smooth[iter_id] = smooth_fun(stat[iter_lw:iter_up])

        sample_smooth[:, stat_id] = stat_smooth

    return sample_smooth


@add_plot_option(option="save")
def measure_plot(measure_dict, time_dict,
                 pal=None, smooth_span=5, smooth_method="mean",
                 xaxis_type="time", ylim=None,
                 legend_loc="upper left", title=None):
    """

    :param measure_dict:
            a dictionary of {method_name: n_iter x 4 summary of target measure}
    :param time_dict:
            a dictionary of {method_name: average step time}
    :param pal: palette, one for each method in the dictionary
    :param smooth_span: num of iterations to include for line smoothing
    :param legend_loc: location of legend 'upper/lower' + 'left/right'
    :param xaxis_type: type for x-axis: iteration or time
    :param ylim: (optional) limit for y-axis
    :param title: (optional) parameter, plot title
    :return:
    """

    # set up environment & device
    if pal is None:
        pal = ["#DC3522", "#11111D", "#0B486D", "#D35400", "#286262"]
    method_list = np.sort(measure_dict.keys())

    # plot
    sns.set_style('darkgrid')
    x_span = np.inf
    for method_id in range(len(method_list)):
        method = method_list[method_id]
        measure_array = measure_dict[method]
        measure_array = \
            iter_smooth(measure_array, span=smooth_span, method=smooth_method)
        if xaxis_type == "iter":
            x_array = np.arange(measure_array.shape[0]-1)
            x_span = np.max(x_array)
        elif xaxis_type == "time":
            step_time = time_dict[method]
            if method == "ADVI":
                step_time = step_time/50
            elif method == "NFVI":
                step_time = step_time/100
            x_array = np.arange(measure_array.shape[0]-1) * step_time
            x_span = np.min((x_span, np.max(x_array)))

        # mean and ci plot
        plt.plot(x_array, measure_array[1:, 2],
                 c=pal[method_id], label=method)
        plt.fill_between(x=x_array,
                         y1=measure_array[1:, 1],
                         y2=measure_array[1:, 3],
                         color=pal[method_id], alpha=0.5)

    # add extra stuff
    plt.legend(loc=legend_loc)
    if xaxis_type == "time":
        plt.xlim((-x_span/100, x_span + x_span/100))
        plt.xlabel("Elapsed Time (seconds)")
    else:
        plt.xlabel("Iterations")

    if title is not None:
        plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)


#####################################
# 3. compute per iteration metric   #
#####################################
# 3.0 prepare model
n = 50
p = 5
k = 2

# construct model object
y_placeholder = theano.shared(np.random.normal(size=(n, p)))
e_placeholder = np.zeros(shape=(n, p))

model = Model(y_placeholder, param_model, e=e_placeholder)


# 3.1 Mean, Cov, Coverage Probability (for stacked factor matrix)
mean_dist = dict()
cov_dist = dict()
prob_dist = dict()
plik_dist = dict()
llik_dist = dict()

# mean_dist = pk.load(open("./result/mean.pkl", "rb"))
# cov_dist = pk.load(open("./result/cov.pkl", "rb"))
# prob_dist = pk.load(open("./result/prob.pkl", "rb"))
# llik_dist = pk.load(open("./result/llik.pkl", "rb"))


for method in method_list:
    # moment measure
    u, v, y, sample = U[method], V[method], Y[method], F[method]

    print("%s: mean" % method)
    mean_dist[method] = moment_measure(sample, type="mean", n_boot=100)
    print("%s: cov" % method)
    cov_dist[method] = moment_measure(sample, type="cov", n_boot=100)
    print("%s: coverage probability" % method)
    prob_dist[method] = moment_measure(sample, type="prob", n_boot=100)
    print("%s: prior log likelihood" % method)
    plik_dist[method] = moment_measure(sample, type="llik_prior", n_boot=100)
    print("%s: posterior log likelihood" % method)
    llik_raw = compute_llik(u, v, y, lik_fun=model.llik_full)
    llik_dist[method] = moment_measure(llik_raw, type="llik", n_boot=100)


pk.dump(mean_dist, open("./result/mean.pkl", "wb"))
pk.dump(cov_dist, open("./result/cov.pkl", "wb"))
pk.dump(prob_dist, open("./result/prob.pkl", "wb"))
pk.dump(plik_dist, open("./result/plik.pkl", "wb"))
pk.dump(llik_dist, open("./result/llik.pkl", "wb"))


measure_plot(
    measure_dict=mean_dist, time_dict=T,
    xaxis_type="iter",
    ylim=(0, 0.1), smooth_span=5, legend_loc="upper right",
    title="Mean Square Error for Prediction",
    save_size=[10, 5], save_addr=report_addr + "/measure/mean_dist.pdf")

measure_plot(
    measure_dict=cov_dist, time_dict=T,
    xaxis_type="iter",
    ylim=(0, 0.5), legend_loc="upper right",
    title="Mean Square Error for Population Covariance ",
    save_size=[10, 5], save_addr=report_addr + "/measure/cov_dist.pdf")

measure_plot(
    measure_dict=prob_dist, time_dict=T,
    xaxis_type="iter",
    ylim=(0.8, 1), legend_loc="lower right",
    title="95% Coverage Probability for U and V",
    save_size=[10, 5], save_addr=report_addr + "/measure/prob_dist.pdf")

measure_plot(
    measure_dict=plik_dist, time_dict=T,
    xaxis_type="iter",
    ylim=(-2, -1), smooth_span=3, legend_loc="lower right",
    title="Prior Log Likelihood",
    save_size=[10, 5], save_addr=report_addr + "/measure/plik_dist.pdf")

measure_plot(
    measure_dict=llik_dist, time_dict=T,
    xaxis_type="iter",
    ylim=(-300, -220), smooth_span=3, legend_loc="lower right",
    title="Posterior Log Likelihood",
    save_size=[10, 5], save_addr=report_addr + "/measure/llik_dist.pdf")



# ##########################################
# # 4. compute per iteration metric, KSD   #
# ##########################################
# n = 50
# p = 5
# k = 2
# mean_true = 0
# sd_true = 0.2
#
# eig_F = pk.load(open("./result/eig_F.pkl", "r"))
# method = "NUTS"
#
# # prepare model and data
# y_placeholder = theano.shared(np.random.normal(size=(n, p)))
# e_placeholder = np.zeros(shape=(n, p))
#
# model = Model(y_placeholder, param_model, e=e_placeholder)
#
# eigen_sample_all = eig_F[method]
#
# #########################################################
# # evaluate convergence sample
#
# pval_iter = dict()
# ksd_iter = dict()
# n_boot = 100
# iter_freq = 10
#
# for method in method_list:
#     print("now evaluating method: %s" % method)
#     eigen_sample_all = eig_F[method]
#     n_iter, n_chain = eigen_sample_all.shape[:2]
#     ksd_iter[method] = np.zeros(shape=(n_iter/iter_freq+1, 4))
#
#     # prepare kernel
#     for iter_id in tqdm(range(0, n_iter, iter_freq)):
#         eigen_sample = eigen_sample_all[iter_id]
#
#         # testing
#         sigma2 = RBF().set_sigma(eigen_sample)
#         rbf = RBF(sigma2=sigma2)
#         ksd = KSD(model=model, kernel=rbf, eigen=True)
#         ksd_val, _ = ksd.stat(eigen_sample)
#
#         # bootstrap confidence interval
#         ksd_ci = np.zeros(shape=n_boot)
#         for b in range(n_boot):
#             boot_id = np.random.choice(n_chain, n_chain)
#             ksd_ci[b] = ksd.stat(eigen_sample[boot_id])[0]
#         ksd_lo, ksd_md, ksd_up = np.percentile(ksd_ci, (5, 50, 95))
#
#         # store
#         ksd_iter[method][iter_id/iter_freq] = \
#             np.array([ksd_val, ksd_lo, ksd_md, ksd_up])
#
# pk.dump(ksd_iter, open("./result/ksd.pkl", "wr"))
#
# #########################################################
# # plot
# ksd_dist = pk.load(open("./result/ksd.pkl", "r"))
# measure_plot(
#     measure_dict=ksd_dist, ylim=(0, 4),
#     smooth_span=10, smooth_method="median",
#     legend_loc="upper right", title="KSD for eigenvalue of U and V",
#     save_size=[12, 5], save_addr=report_addr + "/measure/kern_dist.pdf")
