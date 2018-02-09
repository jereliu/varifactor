from __future__ import division

import numpy as np


# KSD

class KSD:
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

        H = K * S_gram + B + C + K_hess

        # V-statistic
        stat = N * np.mean(H)

        return stat, H

    def pval(self, sample):
        raise NotImplementedError()


if __name__ == "__main__":
    import kgof
    import kgof.data as data
    import kgof.density as density
    import kgof.goftest as gof
    import kgof.kernel as ker
    import kgof.util as util

    import matplotlib
    import matplotlib.pyplot as plt

    import numpy as np
    import scipy.stats as stats

    from varifactor.metric import KSD
    from varifactor.util.kernel import RBF



    seed = 13
    n = 800
    d = 15

    ds = data.DSLaplace(d=d, loc=0, scale=1.0 / np.sqrt(2))
    dat = ds.sample(n, seed=seed + 1)

    p = density.IsotropicNormal(np.zeros(d), 1.0)

    X = dat.data()
    sigma2 = RBF().set_sigma(X)
    rbf = RBF(sigma2=sigma2)


    class tempmd:
        def __init__(self, p):
            self.model = p
            self.model.grad = p.grad_log


    ksd = KSD(model=tempmd(p), kernel=rbf)
    ksd.stat(X)

    # Compare with kgof
    alpha = 0.01
    k = ker.KGauss(sigma2)

    # inverse multiquadric kernel
    # From Gorham & Mackey 2017 (https://arxiv.org/abs/1703.01717)
    # k = ker.KIMQ(b=-0.5, c=1.0)

    bootstrapper = gof.bootstrapper_rademacher
    kstein = gof.KernelSteinTest(p, k, bootstrapper=bootstrapper,
                                 alpha=alpha, n_simulate=500, seed=seed + 1)

    kstein_result = kstein.perform_test(dat, return_simulated_stats=True,
                                        return_ustat_gram=True)
    kstein_result