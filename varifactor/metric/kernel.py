from __future__ import division

import logging

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
    def __init__(self, model, kernel):
        self.model = model.model
        self.kernel = kernel

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
        S = self.model.grad(X)
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

        H = (K * S_gram + B + C + K_hess) / D # standardize by dimension
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
    import kgof.data as data
    import kgof.density as density
    import kgof.goftest as gof

    import numpy as np

    from varifactor.metric.kernel import KSD
    from varifactor.util.kernel import RBF

    from tqdm import tqdm

    seed = 13
    n = 1000
    d = 10

    ds = data.DSLaplace(d=d, loc=0, scale=1.0 / np.sqrt(2))
    dat = ds.sample(n, seed=seed + 1)

    p = density.IsotropicNormal(np.zeros(d), 1.0)

    class tempmd:
        def __init__(self, p):
            self.model = p
            self.model.grad = p.grad_log

    ######################################################################
    # kernel stein test
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

            ksd = KSD(model=tempmd(p), kernel=rbf)
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



    n = 100
    p = 20
    k = 5
    U = np.random.normal(size=(n,k))
    V = np.random.normal(size=(p,k))

    def trA(U, V, A=np.exp):
        return np.sum(A(U.dot(V.T)))

    def dtrA(U, V, eps = 1e-6):
        n, k = U.shape
        p, k = V.shape

        dA = np.zeros((n,k))
        for i in range(n):
            for j in range(k):
                U_add = np.zeros((n,k))
                U_add[i, j] = 1

                U_low = U - U_add * eps
                U_upp = U + U_add * eps
                dA[i, j] = (trA(U_upp, V) - trA(U_low, V))/(eps * 2)
        return dA