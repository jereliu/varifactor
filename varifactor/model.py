from __future__ import division
import logging

import numpy as np
import autograd
import autograd.numpy as atnp

import theano as tt
import pymc3 as pm

from varifactor.util.decorator import defunct

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model")


class NEFactorModel(object):
    def __init__(self, y, param,
                 u=None, v=None, e=None,
                 track_transform=False,
                 parametrization="primal"):
        """

        :param y: a theano.tensor.sharedvar.TensorSharedVariable object, created using theano.shared(np.array)
        :param param:
        :param u: observed value of u, if None then u is treated as an unobserved variable
        :param v: observed value of v, if None then v is treated as an unobserved variable
                    supplied v must match the assumed dimension
        :param e: observed value of e, if None then e is treated as an unobserved variable
        :param track_transform: whether track transformed factors
        """

        # fill in data parameter
        self.family = param.y['family']
        self.n, self.p = y.shape.eval()
        self.k = param.theta['k']

        # fill in observed values
        self.y = y
        self.u = u
        self.v = v
        self.e = e

        # fill in hyper-parameters
        self.eps_sd = param.theta['eps_sd']
        self.u_par = param.u
        self.v_par = param.v

        # specify variance-covariance structure for u,v
        if param.v['cov'] is None:
            self.v_cov = np.eye(self.p)
        else:
            v_cov = param.u['cov']
            assert len(v_cov.shape) == 2 & v_cov.shape[0] == self.p & v_cov.shape[1] == self.p # check dimension
            self.v_cov = v_cov

        # check supplied u, v, and e
        if u is not None:
            if not u.shape == (self.k, self.n):
                raise ValueError(
                    "provided u does not match model dimension (%d, %d), "
                    "either supply new u or change model dimension in param.theta['k']" % (self.k, self.n))
        if v is not None:
            if not v.shape == (self.k, self.p):
                raise ValueError(
                    "provided v does not match model dimension (%d, %d), "
                    "either supply new u or change model dimension in param.theta['k']" % (self.k, self.p))
        if e is not None:
            if not e.shape == (self.n, self.p):
                raise ValueError(
                    "provided e does not match outcome dimension (%d, %d), "
                    "either supply new u or change model dimension in param.theta['k']" % (self.n, self.p))

        # initialize model
        logging.info('\n====================')
        logging.info('initializing NEF factor analysis model..')

        nef_factor = \
            self.parametrize_model(
                parametrization=parametrization,
                track_transform=track_transform)

        logging.info('initialization done')
        logging.info('\n====================')

        self.model = nef_factor


    ##############################################################################
    # method for model definition
    ##############################################################################
    def parametrize_model(self, parametrization="primal", track_transform=False):
        """
        the Primal formulation is the standard formulation using Normal prior
        for factors.

        the Dual formulation avoids rotation ambiguity of factor loading.
        It is achieved by sampling factor loading V as Cholesky decomposition
        of Wishart matrix (using Bartlett decomposition)

        Warning: dual formulation currently not working
        Warning: only applicable to square V matrix
        #TODO: add singular Wishart distribution to allow K<P case

        :param parametrization: primal/dual
        :param track_transform: whether track transformed factors
        :return:
        """
        # parametrization check
        logging.info('parametrization: %s' % parametrization)
        if parametrization == "dual":
            if self.p != self.k:
                raise ValueError("dual formulation requires P = K")

        nef_factor = pm.Model()

        with nef_factor:

            # initialize random variables
            u = pm.Normal(name="U",
                          mu=0, sd=1,
                          shape=(self.k, self.n),
                          observed=self.u,
                          testval=
                              0 * np.random.normal(
                                    loc=0, scale=self.u_par['sd'],
                                    size=(self.k, self.n))
                          )

            # initialize random variables for V, depending on parametrization
            if parametrization == "primal":
                v = pm.Normal(name="V",
                              mu=0, sd=1,
                              shape=(self.k, self.p),
                              observed=self.v,
                              testval=
                                  0 * np.random.normal(
                                        loc=0, scale=self.v_par['sd'],
                                        size=(self.k, self.p))
                              )
            elif parametrization == "dual":
                # a Barlett decomposition of Wishart with
                # degree of freedom = nu and covariance S
                v = pm.WishartBartlett(
                    name="V",
                    S=np.eye(self.p), nu=self.p,
                    testval=np.eye(self.p),
                    return_cholesky=True
                )

            e = pm.Normal(name="e",
                          mu=0, sd=1,
                          shape=(self.n, self.p),
                          observed=self.e,
                          testval=
                              0 * np.random.normal(
                                  loc=0, scale=1,
                                  size=(self.n, self.p))
                          )

            # factor transformation, covariance
            # TODO: allow non-identity covariance for v
            u_t = self.u_par['sd'] * u
            v_t = self.v_par['sd'] * v
            e_t = self.eps_sd * e

            # factor transformation, activation (transpose then apply func to each col)
            if track_transform:
                u_t = pm.Deterministic("Ut", _factor_transform(u_t.T, name=self.u_par["transform"]))
                v_t = pm.Deterministic("Vt", _factor_transform(v_t.T, name=self.v_par["transform"]))
                theta = pm.Deterministic("theta", tt.tensor.dot(u_t, v_t.T) + e_t)
            else:
                u_t = _factor_transform(u_t.T, name=self.u_par["transform"])
                v_t = _factor_transform(v_t.T, name=self.v_par["transform"])
                theta = tt.tensor.dot(u_t, v_t.T) + e_t

            # theta
            y = _nef_family(name="y", theta=theta, n=self.n, p=self.p,
                            family=self.family, observed=self.y)

        return nef_factor

    ##############################################################################
    # method for full distribution: log likelihood
    ##############################################################################

    def llik_full(self, y, u, v, e=None, d=None):
        """
        compute log likelihood wrt full diagonal matrix
        :param y: n x p np.array of observed outcome
        :param u: n x k np.array of latent feature
        :param v: p x k np.array of factor loadings
        :param e: n x p np.array of sampled residuals
        :param d: k x 1 np.array of sampled diagonal element
        :return:
        """
        # TODO: add support for non-identity prior for v
        # TODO: add support for diagonal element d and their prior

        # prepare default parameter matrices
        if e is None:
            e_mat = np.zeros(shape=(self.n, self.p))
        else:
            e_mat = e * self.eps_sd

        if d is None:
            d_mat = np.ones(shape=self.k)
        d_mat = np.diag(d_mat)

        u = u * self.u_par['sd']
        v = v * self.v_par['sd']

        u_mat = _factor_transform(u.T,
                                  name=self.u_par["transform"], format="numpy").T
        v_mat = _factor_transform(v.T,
                                  name=self.v_par["transform"], format="numpy").T

        # construct theta
        theta = reduce(np.dot, [u_mat, d_mat, v_mat.T]) + e_mat

        # construct likelihood
        likelihood = np.sum(y * theta - _nef_partition(theta, family=self.family))

        # construct prior
        prior_u = -0.5 * np.sum(u**2)/(self.u_par['sd']**2)
        prior_v = -0.5 * np.sum(v**2)/(self.v_par['sd']**2)
        prior_all = [prior_u, prior_v]

        # optionally, add d and e to list of priors
        if e is not None:
            prior_e = -0.5 * np.sum(e**2)/(self.eps_sd**2)
            prior_all.append(prior_e)
        if d is not None:
            prior_d = np.nan
            if np.isnan(prior_d):
                raise NotImplementedError("prior for diagonal element d not implemented")
            prior_all.append(prior_d)

        # assemble
        llik_all = likelihood + np.sum(prior_all)

        return llik_all


    ##############################################################################
    # method for eigen distributions: likelihood, log likelihood, gradient
    ##############################################################################

    def lik_eig(self, eigen):
        """
        unnormalized likelihood, wrt the stacked factor matrix F = [U^T, V^T]^T
        :param eigen: a 1 x d numpy array of input
        :return:
        """
        sd = 1
        n = self.n + self.p
        k = self.k

        if not all(eigen[i] > eigen[i + 1] for i in range(len(eigen) - 1)):
            # check if eigen is ordered, if not return 0:
            return 0
        else:
            diff = eigen.reshape((k, 1)) - eigen.T

            lik_eig = np.prod(np.exp(-0.5/(sd**2) * np.sum(eigen))) * \
                        np.prod(eigen)**((n - k - 1)/2) * \
                        np.prod(diff[diff > 0])

        return lik_eig

    def llik_eig(self, eigen):
        """
        log likelihood, wrt the stacked factor matrix F = [U^T, V^T]^T
        :param eigen: a 1 x d numpy array of input
        :return:
        """
        sd = 1
        n = self.n + self.p
        k = self.k

        if not all(eigen[i] > eigen[i + 1] for i in range(len(eigen) - 1)):
            # check if eigen is ordered, if not return NA:
            return np.nan
        else:
            diff = (eigen.reshape(k, 1) - eigen)[np.triu_indices(k, 1)]
            log_diff = atnp.sum(atnp.log(diff))

            log_lik = -0.5 / (sd ** 2) * atnp.sum(eigen) + \
                      0.5 * (n - k - 1) * atnp.sum(atnp.log(eigen)) + log_diff

        return log_lik

    def grad_eig(self, eigen, indep=True, cov=None, native=True):
        """
        assuming prior distribution is multivariate normal,
        produce gradient function with respect to each of the n eigenvalues
        of the stacked factor matrix F = [U^T, V^T]^T

        :param eigen: eigenvalues, a sample_size x k matrix
        :param indep: whether covariance for prior MVN is identity
        :param native: whether do explicit calculation, or use autograd (slower)
        :return: an n x d numpy array of gradients.
        """
        sd = 1
        n = self.n + self.p
        k = self.k
        sample_size = int(eigen.size/k)

        eigen = eigen.reshape((sample_size, k))

        if native:
            # explicit calculation

            # 0. prepare meta data

            # 1. Calculate each component:
            # 1.1 sum of inverse pairwise difference matrix
            # (first compute difference, then take absolute, then inverse & row sum)
            diff_mat = np.apply_along_axis(
                lambda eig: eig.reshape(k, 1) - eig, 1, eigen)
            diff_inv = \
                np.divide(1, diff_mat, where=(diff_mat != 0)) * \
                       (diff_mat != 0)  # extra protection for devision by zero error
            diff_inv_sum = np.sum(diff_inv, axis=-1)

            if indep:
                # produce final result
                grad = -0.5/(sd**2) + 0.5 * (n - k - 1)/eigen + diff_inv_sum
            else:
                # need covariance structure
                if cov is None:
                    raise ValueError("need to supply covariance structure for non indep prior")
                raise NotImplementedError
        else:
            g = autograd.elementwise_grad(self.llik_eig)
            grad = np.array([g(eigen[i, :]) for i in range(sample_size)])

        return grad

    @defunct
    def eigrad_u(self, u, native=False):
        """
        produce gradient function with respect to U (very noisy, due to variation in eigenvectors)
        :param u:
        :param native: whether to use build-in pymc3 function dlogp to calculate gradient
        :return:
        """
        # produce gradient function with respect to U

        # first compute d_L/d_U, a N x K matrix
        if native:
            # use example: grad_fun({'U': u_train.T})
            dL_dU = self.model.dlogp()({'U': u.T})
        else:
            dL_dU = self._grad_LU(u)

        # then compute d_U/d_eig, a rank x N x K matrix
        dU_deig = self._grad_Ud(u)

        # compute product:
        rank = dU_deig.shape[0]
        dL_deig = np.zeros(shape=rank)
        for d in range(rank):
            dL_deig[d] = np.mean(dL_dU * dU_deig[d, :, :])

        return dL_deig

    @defunct
    def _grad_LU(self, u):
        """
        compute d_L/d_U
        :param U: a N x K matrix
        :return:
        """
        # explicit calculation for gradient wrt U
        y = self.y
        v = self.v
        theta = u.dot(v)

        dA = _nef_partition_deriv(theta)
        dL = (y - dA).dot(v.T) - u/self.u_par['sd']**2
        return dL

    @defunct
    def _grad_Ud(self, u):
        """
        compute d_U/d_eig
        :param U: a N x K matrix
        :return:
        """
        # explicit calculation for gradient wrt U
        l, d, r = np.linalg.svd(u, full_matrices=False)

        dU = np.zeros((len(d),) + u.shape)

        for i in range(len(d)):
            dU[i, :, :] += d[i] * l[:, i, np.newaxis].dot(r[i, np.newaxis])

        return dU


##################################
# helper functions ###############
##################################

def _factor_transform(y, name="identity", format=None):
    if name == "identity":
        y_transform = y
    else:
        # else transform using theano tensor operations
        if name == "exp":
            y_transform = pm.math.exp(y)
        elif name == "softplus":
            y_transform = tt.tensor.nnet.softmax(y)
        elif name == "softmax":
            # column-wise softmax
            y_transform = tt.tensor.nnet.softmax(y.T).T
        else:
            raise ValueError('function name (' + str(name) + ') not defined')

        if format is "numpy":
            y_transform = y_transform.eval()

    return y_transform


def _nef_family(theta, n, p, observed, family="Gaussian", name="y"):
    """
    :param theta: value of natural parameter
    :param n: sample size
    :param p: sample dimension
    :param family: distribution name
    :param name: tf name of the output variable
    :return:
    """

    if family == "Gaussian":
        mu_par = theta
        y = pm.Normal(mu=mu_par, sd=1, shape=(n, p),
                      observed=observed, name=name)

    elif family == "Poisson":
        lambda_par = pm.math.exp(theta)
        y = pm.Poisson(mu=lambda_par, shape=(n, p),
                       observed=observed, name=name)

    elif family == "Binomial":
        # TODO: give options to specify binom_n
        n_binom = 10
        logging.warn('n is fixed to ' + str(n_binom) + ' for Binomial(n, p)')

        p_par = 1/(1 + pm.math.exp(-theta))
        y = pm.Binomial(n=n_binom, p=p_par,
                        shape=(n, p), observed=observed,
                        name=name)

    else:
        raise ValueError('distribution family (' + str(family) + ') not supported')

    return y


def _nef_partition(theta, family="Gaussian"):
    """
    :param theta: value of natural parameter
    :param n: sample size
    :param family: distribution name
    :return:
    """

    if family == "Gaussian":
        A = theta**2/2.

    elif family == "Poisson":
        A = np.exp(theta)

    elif family == "Binomial":
        # TODO: give options to specify binom_n
        n_binom = 10
        logging.warn('n is fixed to ' + str(n_binom) + ' for Binomial(n, p)')
        A = n_binom * np.log(1 + np.exp(theta))

    else:
        raise ValueError('distribution family (' + str(family) + ') not supported')

    return A


def _nef_partition_deriv(theta, family="Gaussian"):
    """
    :param theta: value of natural parameter
    :param n: sample size
    :param family: distribution name
    :return:
    """

    if family == "Gaussian":
        deriv = theta

    elif family == "Poisson":
        deriv = np.exp(theta)

    elif family == "Binomial":
        # TODO: give options to specify binom_n
        n_binom = 10
        logging.warn('n is fixed to ' + str(n_binom) + ' for Binomial(n, p)')
        deriv = n_binom * 1/(1 + np.exp(-theta))

    else:
        raise ValueError('distribution family (' + str(family) + ') not supported')

    return deriv


if __name__ == "__main__":
    import pymc3 as pm
    n = 100
    p = 20
    k = 5

    u_obs = np.random.normal(size=(n, k))
    v_obs = np.random.normal(size=(p, k))
    y_obs = u_obs.dot(v_obs.T)

    with pm.Model() as model:
        u = pm.Normal('u', mu=0, sd=1, shape=(n, k))
        theta = u.dot(v_obs.T)
        y = pm.Normal('y', mu=theta, sd=1, shape=(n, p), observed=y_obs)

    grad = model.dlogp()

    u_test = np.random.normal(size=(n, k))
    grad({'u': u_test})
    (y_obs - u_test.dot(v_obs.T)).dot(v_obs) - u_test


    grad2({'U': u_test.T}).reshape(5, 100).T - ((y_obs - u_test.dot(v_obs.T)).dot(v_obs) - u_test * 100)
    grad({'u': u_test}).reshape(100, 5) - ((y_obs - u_test.dot(v_obs.T)).dot(v_obs) - u_test)
