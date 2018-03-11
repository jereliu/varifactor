from __future__ import division

import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metric")


class KernelDistance:
    def __init__(self, model, kernel):
        self.model = model.model
        self.kernel = kernel

    def stat(self, X):
        raise NotImplementedError

    def test(self, X, nboot=1000):
        raise NotImplementedError

# KSD


class KSD(KernelDistance):
    def __init__(self, model, kernel, eigen=False):
        """

        :param model:
        :param kernel:
        :param eigen: if test for eigenvalues of random matrix
        """
        self.model = model.model
        self.kernel = kernel
        self.grad = model.grad_eig if eigen else model.grad

    def stat(self, X):
        """
        Compute the V statistic as in Section 2.2 of Chwialkowski et al., 2016.
        """
        # 0. prepare data
        N, D = X.shape
        kernel = self.kernel

        # logging.info('\n====================')
        # logging.info('computing test statistic')

        # 1. compute kernel and score
        S = self.grad(X)
        K = kernel.eval(X, X)

        # 2. compute score gram and kernel Hessian
        S_gram = S.dot(S.T)
        K_hess = kernel.hess(X)

        B = np.zeros((N, N))
        C = np.zeros((N, N))
        for d in range(D):
            S_d = S[:, d]
            K_grad_x = kernel.grad(X, dim=d, wrt="x")
            K_grad_y = kernel.grad(X, dim=d, wrt="y")

            B += K_grad_x * S_d
            C += (K_grad_y.T * S_d).T

        H = (K * S_gram + B + C + K_hess) / D**2 # standardize by dimension
        # logging.info('computation done')
        # logging.info('\n====================')

        # V-statistic
        stat = N * np.mean(H)

        return stat, H

    def test(self, X, nboot=1000):
        N, D = X.shape
        stat, H = self.stat(X)

        # wild bootstrap
        # logging.info('\n====================')
        # logging.info('bootstrapping')

        boot_sample = np.zeros(nboot)
        for i in range(nboot):
            #w = 2 * np.random.binomial(1, 0.5, N) - 1
            w = 2.0 * np.random.randint(0, 1 + 1, N) - 1.0
            # n * [ (1/n^2) * \sum_i \sum_j h(x_i, x_j) w_i w_j ]
            boot_stat = w.dot(H.dot(w/float(N)))
            # This is a bootstrap version of n*V_n
            boot_sample[i] = boot_stat

        # logging.info('bootstrap done')
        # logging.info('\n====================')

        # approximate p-value with the permutations
        p_value = np.mean(stat < boot_sample)

        return p_value, stat, boot_sample


if __name__ == "__main__":
    import os
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    from tqdm import tqdm

    import numpy as np

    import kgof.data as data
    import kgof.density as density
    import kgof.goftest as gof

    from varifactor.metric.kernel import KSD
    from varifactor.util.kernel import RBF

    ##############################################
    # 1. Unit Test for multivariate Gaussian     #
    ##############################################

    seed = 13
    n = 1000
    d = 10

    ds = data.DSLaplace(d=d, loc=0, scale=1.0 / np.sqrt(2))
    dat = ds.sample(n, seed=seed + 1)

    normal_dist = density.IsotropicNormal(np.zeros(d), 1.0)

    class tempmd:
        def __init__(self, p):
            self.model = p
            self.model.grad = p.grad_log

    ######################################################################
    # kernel stein test, with gaussian distribution
    n_trial = 10
    n_rep = 100

    container = {'ksd': np.zeros(n_trial),
                 'pval': np.zeros(n_trial)}
    for i in range(n_trial):
        container_rep = {'ksd': np.zeros(n_rep),
                         'pval': np.zeros(n_rep)}
        for j in tqdm(range(n_rep)):
            X = np.random.normal(i/100., 1, size=(n, d))

            sigma2 = RBF().set_sigma(X)
            rbf = RBF(sigma2=sigma2)

            ksd = KSD(model=tempmd(normal_dist), kernel=rbf)
            # ksd_val, ksd_H = ksd.stat(X)
            p_val, ksd_val, _ = ksd.test(X)
            container_rep['ksd'][j] = ksd_val
            container_rep['pval'][j] = p_val

        container['ksd'][i] = np.mean(container_rep['ksd'])
        container['pval'][i] = np.mean(container_rep['pval'] < 0.05)

    ######################################################################
    # Compare with kgof
    alpha = 0.01
    # k = ker.KGauss(sigma2)

    # inverse multiquadric kernel
    # From Gorham & Mackey 2017 (https://arxiv.org/abs/1703.01717)
    # k = ker.KIMQ(b=-0.5, c=1.0)

    bootstrapper = gof.bootstrapper_rademacher
    kstein = gof.KernelSteinTest(p, k, bootstrapper=bootstrapper,
                                 alpha=alpha, n_simulate=1000, seed=seed + 1)

    kstein_result = kstein.perform_test(dat, return_simulated_stats=True,
                                        return_ustat_gram=True)
    kstein_result

    ######################################################################
    # compare with kgof.FSSD
    tr, te = dat.split_tr_te(tr_proportion=0.2, seed=2)
    J = 5

    # There are many options for the optimization.
    # Almost all of them have default values.
    # Here, we will list a few to give you a sense of what you can control.
    # Full options can be found in gof.GaussFSSD.optimize_locs_widths(..)
    opts = {
        'reg': 1e-2,  # regularization parameter in the optimization objective
        'max_iter': 50,  # maximum number of gradient ascent iterations
        'tol_fun': 1e-4,  # termination tolerance of the objective
    }

    # make sure to give tr (NOT te).
    # do the optimization with the options in opts.
    V_opt, gw_opt, opt_info = gof.GaussFSSD.optimize_auto_init(p, tr, J, **opts)
    alpha = 0.01
    fssd_opt = gof.GaussFSSD(p, gw_opt, V_opt, alpha)
    test_result = fssd_opt.perform_test(te)
    test_result

    ######################################################################
    # compare with kgof.ME-opt
    import kgof.intertst as tgof
    J = 5

    op = {'n_test_locs': J, 'seed': seed + 5, 'max_iter': 200,
          'batch_proportion': 1.0, 'locs_step_size': 1.0,
          'gwidth_step_size': 0.1, 'tol_fun': 1e-4}
    # optimize on the training set
    me_opt = tgof.GaussMETestOpt(p, n_locs=J, tr_proportion=0.5,
                                 alpha=alpha, seed=seed + 1)

    # Give the ME test the full data. Internally the data are divided into tr and te.
    me_opt_result = me_opt.perform_test(dat, op)
    me_opt_result

    #########################################################
    # 2. Unit Test for eigenvalue of multivariate Gaussian  #
    #########################################################
    import os
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    from tqdm import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt

    import numpy as np
    import scipy.stats as st
    import theano

    from varifactor.model import NEFactorModel as Model
    from varifactor.metric.kernel import KSD
    from varifactor.util.kernel import RBF
    from varifactor.setting import param_model

    def draw_eigen(n, k, sample_size=int(1e3), mean_true=0, sd_true=0.2):
        eigen_sample = np.zeros(shape=(sample_size, k))

        for i in range(sample_size):
            X = np.random.normal(loc=mean_true, scale=sd_true, size=(n, k))
            eigen_sample[i] = np.linalg.svd(X, compute_uv=False) ** 2

        return eigen_sample


    def plot_2dcontour(f=None, data=None, title="", xlim=(1, 3.5), ylim=(1, 2.5)):
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


    #########################################################
    # 2.1 draw samples from eigenvalue distribution of U prior
    sample_size = 500
    n = 100
    p = 15
    k = 2
    mean_true = 0
    sd_true = 0.2

    # construct model
    y_placeholder = theano.shared(np.random.normal(size=(n, p)))
    e_placeholder = np.zeros(shape=(n, p))

    model = Model(y_placeholder, param_model, e=e_placeholder)

    # plot eig density and compare with sample
    eigen_sample = draw_eigen(n, k, sample_size, mean_true, sd_true)
    plot_2dcontour(data=eigen_sample, title="sample likelihood")
    plot_2dcontour(f=model.lik_eig, title="analytical likelihood")
    plot_2dcontour(f=model.llik_eig, title="analytical log likelihood")

    # examine if likelihood / log likelihood implementation are correct
    lik = [model.lik_eig(eigen_sample[i]) for i in range(eigen_sample.shape[0])]
    llik = [model.llik_eig(eigen_sample[i]) for i in range(eigen_sample.shape[0])]

    assert np.sqrt(np.mean((np.log(lik) - llik)**2)) < 1e-10

    # check if derivative is correctly implemented
    grad_native = model.grad_eig(eigen_sample, native=True)
    grad_autogd = model.grad_eig(eigen_sample, native=False)

    assert np.sqrt(np.mean((grad_native - grad_autogd)**2)) < 1e-10

    #########################################################
    # construct KSD test and compute distance, single test run

    sigma2 = RBF().set_sigma(eigen_sample)
    rbf = RBF(sigma2=sigma2)

    ksd = KSD(model=model, kernel=rbf, eigen=True)
    p_val, ksd_val, _ = ksd.test(eigen_sample)

    #########################################################
    # examine power and Type I error of the test, mean-shift alternative

    sample_size = 500
    n_trial = 5
    n_rep = 100

    mean_range = np.linspace(0, 1, num=n_trial)

    container = {'ksd': np.zeros(n_trial),
                 'pval': np.zeros(n_trial)}
    for i in range(n_trial):
        container_rep = {'ksd': np.zeros(n_rep),
                         'pval': np.zeros(n_rep)}
        print("Iteration %d/%d" % (i+1, n_trial))
        for j in tqdm(range(n_rep)):
            mean_true = mean_range[i]
            X = draw_eigen(n, k, sample_size, mean_true, sd_true)

            sigma2 = RBF().set_sigma(X)
            rbf = RBF(sigma2=sigma2)

            ksd = KSD(model=model, kernel=rbf, eigen=True)
            # ksd_val, ksd_H = ksd.stat(X)
            p_val, ksd_val, _ = ksd.test(X)
            container_rep['ksd'][j] = ksd_val
            container_rep['pval'][j] = p_val

        container['ksd'][i] = np.mean(container_rep['ksd'])
        container['pval'][i] = np.mean(container_rep['pval'] < 0.05)

