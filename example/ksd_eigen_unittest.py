import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pickle as pk

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as st
import theano

from varifactor.model import NEFactorModel as Model
from varifactor.metric.kernel import KSD
from varifactor.util.kernel import RBF
from varifactor.util.result_handler import read_npy, get_eigen
from varifactor.setting import param_model

res_addr_0 = "./result/Poisson_n50_p5_k2/"
report_addr = \
    "/home/jeremiah/Dropbox/Research/Harvard/Thesis/Lorenzo/" \
    "1. varifactor/Report/Progress/2018_02_Week_4/plot/"

####################
# 1. read in file  #
####################

method_list = os.listdir(res_addr_0)

U = dict()
V = dict()
eig_S = dict()

sd_u = param_model.u["sd"]
sd_v = param_model.v["sd"]

for method in method_list:
    res_addr = res_addr_0 + "%s/" % (method)
    U[method] = read_npy(res_addr + "U/")
    V[method] = read_npy(res_addr + "V/")
    S_method = np.concatenate((U[method]/sd_u, V[method]/sd_v), axis=-2)
    eig_S[method] = get_eigen(S_method)

del S_method
pk.dump(eig_S, open("./result/eig_S.pkl", "wr"))


#####################################
# 2. helper functions
#####################################


# drawn from true distribution (multivariate normal with sd)
def draw_eigen(n, k, sample_size=int(1e3), mean_true=0, sd_true=1):
    eigen_sample = np.zeros(shape=(sample_size, k))

    for i in range(sample_size):
        X = np.random.normal(loc=mean_true, scale=sd_true, size=(n, k))
        eigen_sample[i] = np.linalg.svd(X, compute_uv=False) ** 2

    return eigen_sample


def plot_2dcontour(f=None, data=None, title="", xlim=(35, 100), ylim=(20, 70)):
    # define grid
    xmin, xmax = xlim
    ymin, ymax = ylim

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
        if not np.isnan(np.max(z)):
            z = z / np.max(z)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.title(title)

    cfset = ax.contourf(xx, yy, z, cmap='Blues')
    return cfset

#####################################
# 3. compute empirical sample      ##
#####################################
N = 50
P = 5
K = 2

eigen_sample = draw_eigen(n=N+P, k=K, sample_size=10000)
dens_plot = \
    sns.jointplot(x=eigen_sample[:,0], y=eigen_sample[:,1], kind="kde",
              xlim=(35, 100), ylim=(20, 70))


dens_plot.savefig(report_addr + "Poisson_n50_p5_k2/eig_true.png")

# drawn from sample distribution
plt.ioff()

eig_S = pk.load(open("./result/eig_S.pkl", "r"))

for method in ["Metropolis", "NUTS", "ADVI"]:
    # create output directory
    out_addr = report_addr + "Poisson_n50_p5_k2/%s/" % (method)

    if not os.path.isdir(out_addr):
        os.mkdir(out_addr)

    # prob container
    eigen_sample = eig_S[method]

    for iter_id in tqdm(range(1, eigen_sample.shape[0], 2)):
        try:
            dens_plot = plot_2dcontour(data=eigen_sample[iter_id])
        except np.linalg.linalg.LinAlgError:
            continue
        plt.savefig(out_addr + "S_%d.png" % (iter_id))

plt.ion()


##############################################
# 4. KSD unit test: type I error and power  ##
##############################################
# prepare model
sample_size = 10000
n = 50
p = 5
k = 2
mean_true = 0
sd_true = 1

y_placeholder = theano.shared(np.random.normal(size=(n, p)))
e_placeholder = np.zeros(shape=(n, p))

model = Model(y_placeholder, param_model, e=e_placeholder)


#########################################################
# 4.1 single test run

# draw eigenvalues and plot kde contour
eigen_sample = draw_eigen(n+p, k, sample_size, mean_true, sd_true)

# plot ideal density
plot_2dcontour(data=eigen_sample[:1000])
plot_2dcontour(f=model.lik_eig)

sigma2 = RBF().set_sigma(eigen_sample)
rbf = RBF(sigma2=sigma2)

ksd = KSD(model=model, kernel=rbf, eigen=True)
p_val, ksd_val, _ = ksd.test(eigen_sample)

#########################################################
# 4.2. Type I error and power, mean-shift alternative

sample_size = 500
n_trial = 5
n_rep = 100
mean_true = 0
sd_true = 1

mean_range = np.linspace(0, 1, num=n_trial)

container = {'ksd': np.zeros(n_trial),
             'pval': np.zeros(n_trial)}
for i in range(n_trial):
    container_rep = {'ksd': np.zeros(n_rep),
                     'pval': np.zeros(n_rep)}
    print("Iteration %d/%d" % (i + 1, n_trial))
    for j in tqdm(range(n_rep)):
        mean_test = mean_range[i]
        X = draw_eigen(n, k, sample_size, mean_test, sd_true)

        sigma2 = RBF().set_sigma(X)
        rbf = RBF(sigma2=sigma2)

        ksd = KSD(model=model, kernel=rbf, eigen=True)
        # ksd_val, ksd_H = ksd.stat(X)
        p_val, ksd_val, _ = ksd.test(X)
        container_rep['ksd'][j] = ksd_val
        container_rep['pval'][j] = p_val

    container['ksd'][i] = np.mean(container_rep['ksd'])
    container['pval'][i] = np.mean(container_rep['pval'] < 0.05)


#########################################################
# 4.3. Type I error and power, , covariance alternative
# distribution of eigenvalue is sensitive to difference in variance

mean_true = 0
sd_true = 1

sample_size = 500
n_trial = 5
n_rep = 100

sd_range = np.linspace(sd_true-0.01, sd_true+0.01, num=n_trial)

container = {'ksd': np.zeros(n_trial),
             'pval': np.zeros(n_trial)}
for i in range(n_trial):
    container_rep = {'ksd': np.zeros(n_rep),
                     'pval': np.zeros(n_rep)}
    print("\n Iteration %d/%d \n" % (i + 1, n_trial))
    for j in tqdm(range(n_rep)):
        sd_test = sd_range[i]
        X = draw_eigen(n, k, sample_size, mean_true, sd_test)

        sigma2 = RBF().set_sigma(X)
        rbf = RBF(sigma2=sigma2)

        ksd = KSD(model=model, kernel=rbf, eigen=True)
        # ksd_val, ksd_H = ksd.stat(X)
        p_val, ksd_val, _ = ksd.test(X)
        container_rep['ksd'][j] = ksd_val
        container_rep['pval'][j] = p_val

    container['ksd'][i] = np.mean(container_rep['ksd'])
    container['pval'][i] = np.mean(container_rep['pval'] < 0.05)


######################################
# 5. apply KSD to simulation result ##
######################################
sample_size = 500
n = 50
p = 5
k = 2
mean_true = 0
sd_true = 0.2

eig_S = pk.load(open("./result/eig_S.pkl", "r"))
method = "NUTS"

# prepare model and data
y_placeholder = theano.shared(np.random.normal(size=(n, p)))
e_placeholder = np.zeros(shape=(n, p))

model = Model(y_placeholder, param_model, e=e_placeholder)

eigen_sample_all = eig_S[method]

#########################################################
# 5.1 trial run on single sample
iter_id = 2000

eigen_sample = eigen_sample_all[iter_id]
plot_2dcontour(data=eigen_sample)
plot_2dcontour(f=model.lik_eig)

# prepare kernel
sigma2 = RBF().set_sigma(eigen_sample)
rbf = RBF(sigma2=sigma2)

# testing
ksd = KSD(model=model, kernel=rbf, eigen=True)
p_val, ksd_val, _ = ksd.test(eigen_sample)

#########################################################
# 5.2 evaluate convergence sample

pval_iter = dict()
ksd_iter = dict()
n_boot = 100
iter_freq = 10

for method in method_list:
    print("now evaluating method: %s" % method)
    eigen_sample_all = eig_S[method]
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
ksd_iter = pk.load(open("./result/ksd.pkl", "r"))


col_list = ["#DC3522", "#11111D", "#0B486D", "#D35400"]
sns.set_style('darkgrid')
for method_id in range(len(method_list)):
    method = method_list[method_id]
    plt.plot(ksd_iter[method][:, 2], c=col_list[method_id])
    plt.fill_between(x=np.arange(ksd_iter[method].shape[0]),
                     y1=ksd_iter[method][:, 1],
                     y2=ksd_iter[method][:, 3],
                     color=col_list[method_id], alpha=0.5)
plt.ylim((0, 4))