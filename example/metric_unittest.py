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

method_list = os.listdir(res_addr)

U = dict()
V = dict()
Y = dict()
F = dict()
eig_F = dict()

if res_addr == "./result/Poisson_n50_p5_k2/":
    sd_u = 0.2
    sd_v = 2
else:
    sd_u = param_model.u["sd"]
    sd_v = param_model.v["sd"]


for method in method_list:
    res_addr = "./result/Poisson_n50_p5_k2/%s/" % (method)
    U[method] = read_npy(res_addr + "U/")
    V[method] = read_npy(res_addr + "V/")
    Y[method] = read_npy(res_addr + "Y/")
    F[method] = np.concatenate((U[method]/sd_u, V[method]/sd_v), axis=-2)
    eig_F[method] = get_eigen(F[method])

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
def measure_plot(measure_dict, pal=None,
                 smooth_span=5, smooth_method="mean",
                 legend_loc="upper left", ylim=None, title=None):
    """

    :param measure_dict:
            a dictionary of methods: n_iter x 4 summary of target measure
    :param pal: palette, one for each method in the dictionary
    :param smooth_span: num of iterations to include for line smoothing
    :param legend_loc: location of legend 'upper/lower' + 'left/right'
    :param ylim: (optional) limit for y-axis
    :param title: (optional) parameter, plot title
    :return:
    """

    # set up environment & device
    if pal is None:
        pal = ["#DC3522", "#11111D", "#0B486D", "#D35400"]
    method_list = np.sort(measure_dict.keys())

    # plot
    sns.set_style('darkgrid')
    for method_id in range(len(method_list)):
        method = method_list[method_id]
        measure_array = measure_dict[method]
        measure_array = \
            iter_smooth(measure_array, span=smooth_span, method=smooth_method)
        # mean and ci plot
        plt.plot(measure_array[1:, 2], c=pal[method_id], label=method)
        plt.fill_between(x=np.arange(measure_array.shape[0]-1),
                         y1=measure_array[1:, 1],
                         y2=measure_array[1:, 3],
                         color=pal[method_id], alpha=0.5)

    # add extra stuff
    plt.legend(loc=legend_loc)
    if title is not None:
        plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)



#####################################
# 3. compute per iteration metric   #
#####################################
# 3.1 Mean, Cov, Coverage Probability (for stacked factor matrix)
mean_dist = dict()
cov_dist = dict()
prob_dist = dict()

for method in method_list:
    sample = F[method]
    mean_dist[method] = moment_measure(sample, type="mean")
    cov_dist[method] = moment_measure(sample, type="cov")
    prob_dist[method] = moment_measure(sample, type="prob")

pk.dump(mean_dist, open("./result/mean.pkl", "wr"))
pk.dump(cov_dist, open("./result/cov.pkl", "wr"))
pk.dump(prob_dist, open("./result/prob.pkl", "wr"))


measure_plot(
    measure_dict=mean_dist, ylim=(0, 0.17), smooth_span=10,
    legend_loc="upper right", title="Posterior Mean",
    save_size=[12, 5], save_addr=report_addr + "/measure/mean_dist.pdf")

measure_plot(
    measure_dict=cov_dist, ylim=(0, 0.5),
    legend_loc="upper right", title="Posterior Covariance",
    save_size=[12, 5], save_addr=report_addr + "/measure/cov_dist.pdf")

measure_plot(
    measure_dict=prob_dist, ylim=(0.8, 1),
    legend_loc="lower right", title="95% Coverage Probability",
    save_size=[12, 5], save_addr=report_addr + "/measure/prob_dist.pdf")

# 3.2. Log-likelihood
n = 50
p = 5
k = 2
mean_true = 0
sd_true = 0.2

# construct model object
y_placeholder = theano.shared(np.random.normal(size=(n, p)))
e_placeholder = np.zeros(shape=(n, p))

model = Model(y_placeholder, param_model, e=e_placeholder)

# extract sample
method = method_list[0]
llik_dist = dict()

for method in method_list:
    u, v, y = U[method], V[method], Y[method]
    llik_raw = compute_llik(u, v, y, lik_fun=model.llik_full)
    llik_dist[method] = moment_measure(llik_raw, type="llik", n_boot=1000)

pk.dump(llik_dist, open("./result/llik.pkl", "wr"))
measure_plot(
    measure_dict=llik_dist,
    legend_loc="lower right", title="Log Likelihood",
    save_size=[12, 5], save_addr = report_addr + "/measure/llik_dist.pdf")



##########################################
# 4. compute per iteration metric, KSD   #
##########################################
n = 50
p = 5
k = 2
mean_true = 0
sd_true = 0.2

eig_F = pk.load(open("./result/eig_F.pkl", "r"))
method = "NUTS"

# prepare model and data
y_placeholder = theano.shared(np.random.normal(size=(n, p)))
e_placeholder = np.zeros(shape=(n, p))

model = Model(y_placeholder, param_model, e=e_placeholder)

eigen_sample_all = eig_F[method]

#########################################################
# evaluate convergence sample

pval_iter = dict()
ksd_iter = dict()
n_boot = 100
iter_freq = 10

for method in method_list:
    print("now evaluating method: %s" % method)
    eigen_sample_all = eig_F[method]
    n_iter, n_chain = eigen_sample_all.shape[:2]
    ksd_iter[method] = np.zeros(shape=(n_iter/iter_freq+1, 4))

    # prepare kernel
    for iter_id in tqdm(range(0, n_iter, iter_freq)):
        eigen_sample = eigen_sample_all[iter_id]

        # testing
        sigma2 = RBF().set_sigma(eigen_sample)
        rbf = RBF(sigma2=sigma2)
        ksd = KSD(model=model, kernel=rbf, eigen=True)
        ksd_val, _ = ksd.stat(eigen_sample)

        # bootstrap confidence interval
        ksd_ci = np.zeros(shape=n_boot)
        for b in range(n_boot):
            boot_id = np.random.choice(n_chain, n_chain)
            ksd_ci[b] = ksd.stat(eigen_sample[boot_id])[0]
        ksd_lo, ksd_md, ksd_up = np.percentile(ksd_ci, (5, 50, 95))

        # store
        ksd_iter[method][iter_id/iter_freq] = \
            np.array([ksd_val, ksd_lo, ksd_md, ksd_up])

pk.dump(ksd_iter, open("./result/ksd.pkl", "wr"))

#########################################################
# plot
ksd_dist = pk.load(open("./result/ksd.pkl", "r"))
measure_plot(
    measure_dict=ksd_dist, ylim=(0, 4),
    smooth_span=10, smooth_method="median",
    legend_loc="upper right", title="KSD for eigenvalue of F",
    save_size=[12, 5], save_addr=report_addr + "/measure/kern_dist.pdf")
