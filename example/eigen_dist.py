import os

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import numpy as np

from varifactor.util.result_handler import read_npy, get_eigen


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
    U[method] = read_npy(res_addr + "U/")
    V[method] = read_npy(res_addr + "V/")

#####################################
# 2. helper functions
#####################################

# drawn from true distribution (multivariate normal with sd)
def draw_eigen(N, K, sample_size = int(1e3), sd_true=2):
    eigen_sample = np.zeros(shape=(sample_size, K))
    for i in tqdm(range(sample_size)):
        eigen_sample[i] = \
            np.linalg.svd(np.random.normal(scale=sd_true, size=(N, K)), compute_uv=False) ** 2
    return eigen_sample


#####################################
# 3. compute empirical sample      ##
#####################################
P = 5
K = 2

eigen_sample = draw_eigen(N=P, K=K, sample_size=10000)
dens_plot = \
    sns.jointplot(x=eigen_sample[:,0], y=eigen_sample[:,1], kind="kde",
              xlim=(-5, 70), ylim=(-5, 30))


dens_plot.savefig(report_addr + "Poisson_n50_p5_k2/V_eig_true.png")

# drawn from sample distribution
plt.ioff()

for method in ["Metropolis", "NUTS", "ADVI"]:
    # create output directory
    out_addr = report_addr + "Poisson_n50_p5_k2/%s/" % (method)

    if not os.path.isdir(out_addr):
        os.mkdir(out_addr)

    # prob container
    eigen_sample = get_eigen(V[method])

    for iter_id in tqdm(range(1, eigen_sample.shape[0], 2)):
        dens_plot = \
            sns.jointplot(x=eigen_sample[iter_id, :, 0],
                          y=eigen_sample[iter_id, :, 1],
                          kind="kde",
                          xlim=(-5, 70), ylim=(-5, 30))
        dens_plot.savefig(out_addr + "V_%d.png" % (iter_id))

plt.ion()

##############################################
# 4. KSD unit test: type I error and power  ##
##############################################

