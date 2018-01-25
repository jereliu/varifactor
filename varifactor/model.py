import logging
import numpy as np
import pymc3 as pm
import theano.tensor as tensor


# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model")


class NEFactorModel(object):
    def __init__(self, y, param):

        # fill in parameter
        self.family = param.y['family']
        self.n, self.p = y.shape
        self.k = param.theta['k']
        self.eps_sd = param.theta['eps_sd']
        self.u_var = param.u['var']
        self.v_var = param.v['var']
        self.u_trans = param.u['transform']
        self.v_trans = param.v['transform']

        if param.u['cov'] is None:
            self.v_cov = np.eye(self.p)
        else:
            v_cov = param.u['cov']
            assert len(v_cov.shape) == 2 & v_cov.shape[0] == self.p & v_cov.shape[1] == self.p # check dimension
            self.v_cov = v_cov

        # initialize model
        nef_factor = pm.Model()
        logging.info('\n====================')
        logging.info('initializing NEF factor analysis model..')

        with nef_factor:

            # initialize random variables
            u = pm.MvNormal(name="U", mu=0, cov=self.u_var * np.eye(self.n),
                            shape=(self.k, self.n),
                            testval=np.random.normal(
                                loc=0, scale=np.sqrt(self.u_var), size=(self.k, self.n))
                            )
            v = pm.MvNormal(name="V", mu=0, cov=self.u_var * self.v_cov,
                            shape=(self.k, self.p),
                            testval=np.random.normal(
                                loc=0, scale=np.sqrt(self.v_var), size=(self.k, self.p))
                            )

            e = pm.Normal(name="e", mu=0, sd=1, shape=(self.n, self.p),
                          testval=np.random.normal(0, self.eps_sd, size=(self.n, self.p)))

            # factor transformation, transpose then apply activation func to each column
            u_t = pm.Deterministic("Ut", _factor_transform(u.T, name=self.u_trans))
            v_t = pm.Deterministic("Vt", _factor_transform(v.T, name=self.v_trans))

            # theta
            theta = pm.Deterministic("theta", tensor.dot(u_t, v_t.T) + e)
            y = _nef_family(name="y", theta=theta, n=self.n, p=self.p,
                            family=self.family, observed=y)

        logging.info('initialization done')
        logging.info('\n====================')
        self.model = nef_factor


# helper functions
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
        mu_par = - theta / 2
        y = pm.Normal(mu=mu_par, sd=np.ones((n, p)), shape=(n, p),
                      observed=observed, name=name + "_gaussian")

    elif family == "Poisson":
        lambda_par = pm.math.exp(theta)
        y = pm.Poisson(mu=lambda_par, shape=(n, p),
                       observed=observed, name=name + "_poisson")

    elif family == "Binomial":
        # TODO: give options to specify binom_n
        n_binom = 10
        logging.warn('n is fixed to ' + str(n_binom) + ' for Binomial(n, p)')

        p_par = 1/(1 + pm.math.exp(-theta))
        y = pm.Binomial(n=n_binom, p=p_par,
                        shape=(n, p), observed=observed,
                        name=name + "_binom_" + str(n_binom))

    else:
        raise ValueError('distribution family (' + str(family) + ') not supported')

    return y
