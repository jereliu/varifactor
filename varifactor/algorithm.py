import tensorflow as tf
import edward as ed


def mfvi(y, u, v, e, y_data, K,
         n_iter=1000, n_samples=50):
    N, P = y_data.shape
    with tf.name_scope("posterior_mfvi"):
        qu = ed.models.Normal(loc=tf.Variable(tf.random_normal([N, K]), name="qu/loc"),
                              scale=tf.nn.softplus(
                                  tf.Variable(tf.random_normal([N, K]), name="qu/scale")),
                              name="qu"
                              )
        qv = ed.models.Normal(loc=tf.Variable(tf.random_normal([P, K]), name="qv/loc"),
                              scale=tf.nn.softplus(
                                  tf.Variable(tf.random_normal([P, K]), name="qv/scale")),
                              name="qv"
                              )
        qe = ed.models.Normal(loc=tf.Variable(tf.random_normal([N, P]), name="qe/loc"),
                              scale=tf.nn.softplus(
                                  tf.Variable(tf.random_normal([N, P]), name="qe/scale")),
                              name="qe"
                              )

    inference = ed.KLqp({u: qu, v: qv, e: qe}, data={y: y_data})
    inference.initialize(n_iter = n_iter, n_samples = n_samples)

    return inference


def hmc(y, u, v, e, y_data, K,
        n_iter = 500, step_size = 0.25, n_steps = 10):
    """
    :param y:
    :param u:
    :param v:
    :param e:
    :param y_data:
    :param K:
    :param T: number of empirical samples
    :return:
    """
    N, P = y_data.shape
    with tf.name_scope("posterior_hmc"):
        qu = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, N, K]), name="qu"))
        qv = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, P, K]), name="qu"))
        qe = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, N, P]), name="qe"))

    inference = ed.HMC({u: qu, v: qv, e: qe}, data={y: y_data})
    inference.initialize(step_size = step_size, n_steps=n_steps)

    return inference


def mh(y, u, v, e, y_data, K, n_iter = 500, step_size = 0.1):
    """
    :param y:
    :param u:
    :param v:
    :param e:
    :param y_data:
    :param K:
    :param T: number of empirical samples
    :return:
    """
    N, P = y_data.shape
    with tf.name_scope("posterior_mh"):
        qu = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, N, K]), name="qu"))
        qv = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, P, K]), name="qu"))
        qe = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, N, P]), name="qe"))

    with tf.name_scope("proposal_mh"):
        pu = ed.models.Normal(loc=u, scale= step_size * tf.ones([N, K]))
        pv = ed.models.Normal(loc=v, scale= step_size * tf.ones([P, K]))
        pe = ed.models.Normal(loc=e, scale= step_size * tf.ones([N, P]))

    inference = ed.MetropolisHastings(
        {u: qu, v: qv, e: qe}, {u: pu, v: pv, e: pe}, data={y: y_data})
    inference.initialize()

    return inference
