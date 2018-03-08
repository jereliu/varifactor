import os

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import numpy as np

from varifactor.util.result_handler import get_npy


res_addr = "./result/Poisson_n50_p5_k2/"
report_addr = \
    "/home/jeremiah/Dropbox/Research/Harvard/Thesis/Lorenzo/" \
    "1. varifactor/Report/Progress/2018_02_Week_4/plot/"

####################
# 1. read in file  #
####################

method_list = os.listdir(res_addr)

U = dict()
V = dict()

for method in method_list:
    res_addr = "./result/Poisson_n50_p5_k2/%s/" % (method)
    U[method] = get_npy(res_addr + "U/")
    V[method] = get_npy(res_addr + "V/")

#####################################
# 2. compute empirical samples      #
#####################################
P = 5
K = 2

# drawn from true distribution
sample_size = int(1e3)
eigen_sample = \
    np.array([
        np.linalg.svd(np.random.normal(scale=2, size=(P, K)), compute_uv=False)**2
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

    for iter_id in tqdm(range(1, n_iter, 2)):
        eigen_sample = \
            np.array([np.linalg.svd(V[method][sample_id, iter_id], compute_uv=False)**2
                        for sample_id in range(n_sample)])
        dens_plot = \
            sns.jointplot(x=eigen_sample[:, 0], y=eigen_sample[:, 1], kind="kde",
                          xlim = (1, 10), ylim = (0, 7))
        dens_plot.savefig(out_addr + "V_%d.png" % (iter_id))

plt.ion()
