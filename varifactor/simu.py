import numpy as np


def data(n, p, k, eps_sd=0.1,
         family="Gaussian",
         uv_scale=[0.1, 0.1],
         uv_trans=["identity", "identity"]):
    """
    construct model
    :param n: sample size
    :param p: sample dimension
    :param k: latent dimension
    :param eps_sd: parameter noise
    :param family: Y distribution
    :param uv_scale: scale for latent factor U & V
    :param uv_trans: nonlinear transformation for latent factors U & V
    :return:
    """

    # generate original factor using numpy
    u = np.random.normal(0, uv_scale[0], size=(n, k))
    v = np.random.normal(0, uv_scale[1], size=(p, k))
    eps = np.random.normal(0, eps_sd, size=(n, p))

    # add correlation structure
    # TODO

    # assemble
    u = _factor_transform(u, name=uv_trans[0])
    v = _factor_transform(v, name=uv_trans[1])
    theta = np.dot(u, v.T)

    # generate observation of natural exponential family
    y = _random_nef(theta + eps, family=family)

    return y, u, v, eps


def _factor_transform(y, name="identity"):
    if name == "identity":
        return y
    else:
        if name == "exp":
            return np.exp(y)
        elif name == "softmax":
            y_out = np.zeros(shape = y.shape)
            for j in range(y.shape[1]):
                e_j = np.exp(y[:, j] - np.max(Y[:, j]))
                y_out[:, j] = e_j / e_j.sum()
            return y_out
        elif name == "softplus":
            return np.exp(y)/(np.exp(y) + 1)
        else:
            raise ValueError('function name (' + str(name) + ') not defined')


def _random_nef(theta, family="Gaussian"):
    """
    random variable generation from natural exponential family (nef)
    :param theta:
    :param family: name of distribution family
    :return:
    """
    n, p = theta.shape
    y = np.zeros((n, p))

    if family == "Gaussian":
        mu_par = -theta / 2
        for d in range(p):
            for n in range(n):
                y[n, d] = np.random.normal(mu_par[n, d], 1)
    elif family == "Poisson":
        lambda_par = np.exp(theta)
        for d in range(p):
            for n in range(n):
                y[n, d] = np.random.poisson(lambda_par[n, d])
    else:
        raise ValueError('distribution (' + str(family) + ') not defined')

    return y
