import tensorflow as tf
import edward as ed
from edward.models import Normal, Poisson

def nef_factor(
        N, P, K, sigma = 0.1,
        family = "Gaussian",
        uv_scale = [0.1, 0.1],
        uv_trans = ["identity", "identity"]):

    u = Normal(loc=tf.zeros([N, K]), scale= uv_scale[0] * tf.ones([N, K]), name = "u")
    v = Normal(loc=tf.zeros([P, K]), scale= uv_scale[1] * tf.ones([P, K]), name = "v")
    e = Normal(loc=tf.zeros([N, P]), scale= sigma * tf.ones([N, P]), name = "e")

    u_t = _factor_transform(u, name = uv_trans[0])
    v_t = _factor_transform(v, name = uv_trans[0])

    theta = tf.matmul(u_t, v_t, transpose_b=True) + e
    y = _nef_family(theta, N, P, family = family, name = "y")

    return y, u, v, e




def _factor_transform(u, name = "identity"):
    if name == "identity":
        return u
    else:
        if name == "exp":
            return tf.exp(u)
        elif name == "softmax":
            return tf.nn.softmax(u, 1)
        elif name == "softplus":
            return tf.nn.softplus(u, 1)
        else:
            raise ValueError("function name '"+ str(name) + "' not defined.")


def _nef_family(theta, N, P, family = "Gaussian", name = "y"):
    """
    :param theta: value of natural parameter
    :param N: sample size
    :param P: sample dimension
    :param dist: distribution name
    :param name: tf name of the output variable
    :return:
    """
    if family == "Gaussian":
        Mu = -theta/2
        y = Normal(loc=Mu, scale=tf.ones([N, P]), name = name + "_gaussian")

    elif family == "Poisson":
        Lambda = tf.exp(theta)
        y = Poisson(rate=Lambda, name = name + "_poisson")
    else:
        raise ValueError('distribution family (' + str(family) + ') not defined')

    return y