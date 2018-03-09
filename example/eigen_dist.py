import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

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
def draw_eigen(n, k, sample_size=int(1e3), mean_true=0, sd_true=0.2):
    eigen_sample = np.zeros(shape=(sample_size, k))

    for i in tqdm(range(sample_size)):
        X = np.random.normal(loc=mean_true, scale=sd_true, size=(n, k))
        eigen_sample[i] = np.linalg.svd(X, compute_uv=False) ** 2

    return eigen_sample


def plot_2dcontour(f=None, data=None):
    # define grid
    xmin, xmax = 2, 6
    ymin, ymax = 2, 6

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # if no function, do kernel density estimation based on data
    if f is None:
        x = data[:, 0]
        y = data[:, 1]
        values = np.vstack([x, y])
        f = st.gaussian_kde(values)
        z = np.reshape(f(positions).T, xx.shape)
    else:
        z = np.array([f(positions[:, i]) for i in range(positions.shape[1])])
        z = np.reshape(z, xx.shape)
        z = z/np.max(z)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    cfset = ax.contourf(xx, yy, z, cmap='Blues')


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

import numpy as np
import scipy.stats as st
import theano

import kgof.data as data
import kgof.density as density
import kgof.goftest as gof

from varifactor.model import NEFactorModel as Model
from varifactor.metric.kernel import KSD
from varifactor.util.kernel import RBF
from varifactor.setting import param_model

# prepare model
sample_size = 10000
n = 100
p = 15
k = 2
mean_true = 0
sd_true = 0.2

y_placeholder = theano.shared(np.random.normal(size=(n, p)))
e_placeholder = np.zeros(shape=(n, p))

model = Model(y_placeholder, param_model, e=e_placeholder)

# draw eigenvalues and plot kde contour
eigen = draw_eigen(n, k, sample_size, mean_true, sd_true)

# plot ideal density
plot_2dcontour(data=eigen)
plot_2dcontour(f=model.lik_eig)


