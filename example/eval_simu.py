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
from varifactor.setting import param_model

####################
# 1. read in file  #
####################

res_addr = "./result/Poisson_n50_p5_k2/"
report_addr = \
    "/home/jeremiah/Dropbox/Research/Harvard/Thesis/Lorenzo/" \
    "1. varifactor/Report/Progress/2018_02_Week_4/plot/"

method_list = os.listdir(res_addr)

U = dict()
V = dict()
S = dict()
eig_S = dict()

sd_u = param_model.u["sd"]
sd_v = param_model.v["sd"]

for method in method_list:
    res_addr = "./result/Poisson_n50_p5_k2/%s/" % (method)
    U[method] = read_npy(res_addr + "U/")
    V[method] = read_npy(res_addr + "V/")
    S[method] = np.concatenate((U[method]/sd_u, V[method]/sd_v), axis=-2)
    eig_S[method] = get_eigen(S[method])

pk.dump(eig_S, open("./result/eig_S.pkl", "wr"))


#####################################
# 2. helper functions               #
#####################################

def emp_distance(sample, type = "mean", sd_true=1):
    n_sample, n_iter, dim1, dim2 = sample.shape

    if type == "mean":
        moment_true = np.zeros(shape=(dim1, dim2))
        moment_sample = np.mean(sample, axis=0)
    elif type == "cov":
        moment_true = np.eye(dim2) * sd_true**2
        sample_temp = np.swapaxes(sample, 0, 1).reshape(n_iter, n_sample*dim1, dim2)
        moment_sample = np.array([np.cov(sample_temp[i].T) for i in range(n_iter)])
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

    dist_iter = np.sqrt(np.mean((moment_sample - moment_true) ** 2, axis=(-2, -1)))

    return dist_iter


def emp_measure(sample, type="mean", n_boot=500, sd_true=1):
    """
    compute empirical measures as well as bootstrap percentiles (5%, 50%, 95%)
    :param sample:
    :param type:
    :param n_boot:
    :param sd_true:
    :return:
    """
    n_chain, n_iter, dim1, dim2 = sample.shape

    # compute mean
    measure_mean = emp_distance(sample, type=type, sd_true=sd_true)

    # compute bootstrap sample
    measure_boot_sample = np.zeros(shape=(n_boot, n_iter))
    for boot_id in tqdm(range(n_boot)):
        boot_chain_id = np.random.choice(n_chain, n_chain)
        measure_boot_sample[boot_id] = \
            emp_distance(sample[boot_chain_id], type=type, sd_true=sd_true)

    measure_pc = np.percentile(measure_boot_sample, (5, 50, 95), axis=0).T
    measure_all = \
        np.concatenate((np.atleast_2d(measure_mean).T, measure_pc), axis=1)

    return measure_all

#####################################
# 3. compute per iteration metric   #
#####################################


# 3.1 Mean, Cov, Coverage Probability (V)
mean_dist = dict()
cov_dist = dict()
prob_dist = dict()

for method in method_list:
    sample = S[method]
    mean_dist[method] = emp_measure(sample, type="mean")
    cov_dist[method] = emp_distance(sample, type="cov")
    prob_dist[method] = emp_distance(sample, type="prob")

pk.dump(mean_dist, open("./result/mean.pkl", "wr"))
pk.dump(cov_dist, open("./result/cov.pkl", "wr"))
pk.dump(prob_dist, open("./result/prob.pkl", "wr"))


# sns.set(style='darkgrid')
# plt.plot(mean_dist['ADVI'][1:])
# plt.plot(mean_dist['NUTS'][1:])
# plt.plot(mean_dist['Metropolis'][1:])
# plt.show()
#
# plt.plot(cov_dist['ADVI'][1:])
# plt.plot(cov_dist['NUTS'][1:])
# plt.plot(cov_dist['Metropolis'][1:])
# plt.ylim((0, 0.2))
# plt.show()
#
# plt.plot(prob_dist['ADVI'][1:])
# plt.plot(prob_dist['NUTS'][1:])
# plt.plot(prob_dist['Metropolis'][1:])
# plt.plot([0, 2000], [0.95, 0.95], "--", color="black")
# plt.ylim((0.8, 1))
# plt.show()

# 3.2. Log-likelihood
# TODO


# ##########################################
# # 4. compute per iteration metric, KSD   #
# ##########################################
# n = 50
# p = 5
# k = 2
# mean_true = 0
# sd_true = 0.2
#
# eig_S = pk.load(open("./result/eig_S.pkl", "r"))
# method = "NUTS"
#
# # prepare model and data
# y_placeholder = theano.shared(np.random.normal(size=(n, p)))
# e_placeholder = np.zeros(shape=(n, p))
#
# model = Model(y_placeholder, param_model, e=e_placeholder)
#
# eigen_sample_all = eig_S[method]
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
#     eigen_sample_all = eig_S[method]
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
#
# col_list = ["#DC3522", "#11111D", "#0B486D", "#D35400"]
# sns.set_style('darkgrid')
# for method_id in range(len(method_list)):
#     method = method_list[method_id]
#     plt.plot(ksd_iter[method][:, 2], c=col_list[method_id])
#     plt.fill_between(x=np.arange(ksd_iter[method].shape[0]),
#                      y1=ksd_iter[method][:, 1],
#                      y2=ksd_iter[method][:, 3],
#                      color=col_list[method_id], alpha=0.5)
# plt.ylim((0, 4))