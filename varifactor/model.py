import logging
import numpy as np
import pymc3 as pm
import theano.tensor as tensor


# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model")


class NEFactorModel(object):
    def __init__(self, y, param,
                 u=None, v=None, e=None, track_transform=False):
        """

        :param y:
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

        nef_factor = pm.Model()
        logging.info('\n====================')
        logging.info('initializing NEF factor analysis model..')

        with nef_factor:

            # initialize random variables
            u = pm.Normal(name="U",
                          mu=0, sd=self.u_par['sd'],
                          shape=(self.k, self.n), observed=u,
                          testval=
                          0 * np.random.normal(
                                loc=0, scale=self.u_par['sd'],
                                size=(self.k, self.n))
                          )
            v = pm.Normal(name="V",
                          mu=0, sd=self.v_par['sd'],
                          shape=(self.k, self.p), observed=v,
                          testval=
                          0 * np.random.normal(
                                loc=0, scale=self.v_par['sd'],
                                size=(self.k, self.p)),
                          )
            e = pm.Normal(name="e",
                          mu=0, sd=1,
                          shape=(self.n, self.p), observed=e,
                          testval=
                          0 * np.random.normal(
                              loc=0, scale=self.eps_sd,
                              size=(self.n, self.p))
                          )

            # factor transformation, transpose then apply activation func to each column
            if track_transform:
                u_t = pm.Deterministic("Ut", _factor_transform(u.T, name=self.u_par["transform"]))
                v_t = pm.Deterministic("Vt", _factor_transform(v.T, name=self.v_par["transform"]))
                theta = pm.Deterministic("theta", tensor.dot(u_t, v_t.T) + e)
            else:
                u_t = _factor_transform(u.T, name=self.u_par["transform"])
                v_t = _factor_transform(v.T, name=self.v_par["transform"])
                theta = tensor.dot(u_t, v_t.T) + e

            # theta
            y = _nef_family(name="y", theta=theta, n=self.n, p=self.p,
                            family=self.family, observed=y)

        logging.info('initialization done')
        logging.info('\n====================')

        self.model = nef_factor

    def eigrad(self, u, native=False):
        """
        produce gradient function with respect to U
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

    def _grad_LU(self, u):
        """
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

    def _grad_Ud(self, u):
        """
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

def _factor_transform(y, name="identity"):
    if name == "identity":
        return y
    else:
        if name == "exp":
            return pm.math.exp(y)
        elif name == "softplus":
            return tensor.nnet.softmax(y)
        elif name == "softmax":
            # column-wise softmax
            return tensor.nnet.softmax(y.T).T
        else:
            raise ValueError('function name (' + str(name) + ') not defined')


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
