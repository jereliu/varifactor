import os

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import numpy as np
from scipy.stats import norm

from varifactor.util.result_handler import get_npy

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

for method in method_list:
    res_addr = "./result/Poisson_n50_p5_k2/%s/" % (method)
    U[method] = get_npy(res_addr + "U/")
    V[method] = get_npy(res_addr + "V/")


#####################################
# 2. helper functions               #
#####################################

def emp_distance(sample, type = "mean", sd_true = 2):
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
        for iter_id in tqdm(range(n_iter)):
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

#####################################
# 3. compute per iteration metric   #
#####################################

# 2.1 Mean, Cov, Coverage Probability (V)
mean_dist = dict()
cov_dist = dict()
prob_dist = dict()

par_list = U

for method in method_list:
    sample = par_list[method]
    mean_dist[method] = emp_distance(sample, type = "mean")
    cov_dist[method] = emp_distance(sample, type = "cov", sd_true = 0.2)
    prob_dist[method] = emp_distance(sample, type = "prob", sd_true = 0.2)

sns.set(style='darkgrid')
plt.plot(mean_dist['ADVI'][1:])
plt.plot(mean_dist['NUTS'][1:])
plt.plot(mean_dist['Metropolis'][1:])
plt.show()


plt.plot(cov_dist['ADVI'][1:])
plt.plot(cov_dist['NUTS'][1:])
plt.plot(cov_dist['Metropolis'][1:])
plt.ylim((0, 0.05))
plt.show()

sns.set(style='darkgrid')
plt.plot(prob_dist['ADVI'][1:])
plt.plot(prob_dist['NUTS'][1:])
plt.plot(prob_dist['Metropolis'][1:])
plt.plot([0, 2000], [0.95, 0.95], "--", color="black")
plt.ylim((0, 1))
plt.show()

# 2.2. Log-likelihood


# 2.3. Eigenvalue KSD
import numpy as np

P = 5
K = 2

# drawn from true distribution
sample_size = int(1e4)
eigen_sample = \
    np.array([
        np.linalg.svd(np.random.normal(scale=2, size=(P, K)), compute_uv=False)
        for i in range(sample_size)])

dens_plot = \
    sns.jointplot(x=eigen_sample[:,0], y=eigen_sample[:,1], kind="kde",
              xlim=(1, 10), ylim=(0, 7))
dens_plot.savefig(report_addr + "Poisson_n50_p5_k2/V_eig_true.png")

# drawn from sample distribution

plt.ioff()

for method in ["Metropolis", "NUTS", "ADVI"]:
    # create output directory
    out_addr = report_addr + "Poisson_n50_p5_k2/%s/" % (method)

    if not os.path.isdir(out_addr):
        os.mkdir(out_addr)

    # prob container
    n_sample, n_iter, P, K = V[method].shape

    for iter_id in tqdm(range(1, n_iter, 10)):
        eigen_sample = \
            np.array([np.linalg.svd(V[method][sample_id, iter_id], compute_uv=False)
                    for sample_id in range(n_sample)])
        dens_plot = \
            sns.jointplot(x=eigen_sample[:, 0], y=eigen_sample[:, 1], kind="kde",
                          xlim = (1, 10), ylim = (0, 7))
        dens_plot.savefig(out_addr + "V_%d.png" % (iter_id))

plt.ion()
