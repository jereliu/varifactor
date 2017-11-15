import numpy as np

def data(N, P, K, sigma = 0.1,
         family = "Gaussian",
         uv_scale = [0.1, 0.1],
         uv_trans = ["identity", "identity"]):
    """
    construct model
    :param N: sample size
    :param P: sample dimension
    :param K: latent dimension
    :param sigma: parameter noise
    :param family: Y distribution
    :param uv_scale: scale for latent factor U & V
    :param uv_trans: nonlinear transformation for latent factors U & V
    :return:
    """

    # generate original factor using numpy
    U = np.random.normal(0, uv_scale[0], size = (N, K))
    V = np.random.normal(0, uv_scale[1], size = (P, K))
    eps = np.random.normal(0, sigma, size = (N, P))

    # add correlation structure
    # TODO

    # assemble
    U = _factor_transform(U, name = uv_trans[0])
    V = _factor_transform(V, name = uv_trans[1])
    Theta = np.dot(U, V.T)

    # generate observation of natural exponential family
    Y = _random_nef(Theta + eps, family = family)

    return Y, U, V, eps



def _factor_transform(Y, name = "identity"):
    if name == "identity":
        return Y
    else:
        if name == "exp":
            return np.exp(Y)
        elif name == "softmax":
            Y_out = np.zeros(shape = Y.shape)
            for j in range(Y.shape[1]):
                e_j = np.exp(Y[:, j] - np.max(Y[:, j]))
                Y_out[:, j] = e_j / e_j.sum()
            return Y_out
        elif name == "softplus":
            return np.exp(Y)/(np.exp(Y) + 1)
        else:
            raise ValueError('function name (' + str(name) + ') not defined')


def _random_nef(Theta, family = "Gaussian"):
    """
    random variable generation from natural exponential family (nef)
    :param theta:
    :param family: name of distribution family
    :return:
    """
    N, P = Theta.shape
    Y = np.zeros((N, P))

    if family == "Gaussian":
        Mu = -Theta/2
        for d in range(P):
            for n in range(N):
                Y[n, d] = np.random.normal(Mu[n, d], 1)
    elif family == "Poisson":
        Lambda = np.exp(Theta)
        for d in range(P):
            for n in range(N):
                Y[n, d] = np.random.poisson(Lambda[n, d])
    else:
        raise ValueError('distribution (' + str(family) + ') not defined')

    return Y
