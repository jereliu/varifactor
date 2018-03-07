import tensorflow as tf
import edward as ed


class algorithm:
    def __init__(self, y, u, v, e, data, K):
        self.y = y
        self.u = u
        self.v = v
        self.e = e
        self.data = data
        self.N = data.shape[0]
        self.P = data.shape[1]
        self.K = K

    def get_sample(self):
        u = self.u.params.eval()
        v = self.v.params.eval()
        e = self.e.params.eval()

        return u, v, e


class mfvi(algorithm):
    def __init__(self, y, u, v, e, data, K):
        self.y = y
        self.u = u
        self.v = v
        self.e = e
        self.data = data
        self.N = data.shape[0]
        self.P = data.shape[1]
        self.K = K

    def initialize(self, n_iter=1000, n_samples=50):
        with tf.name_scope("posterior_mfvi"):
            self.qu = \
                ed.models.Normal(
                    loc=tf.Variable(tf.random_normal([self.N, self.K]),
                                    name="qu/loc"),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.N, self.K]), name="qu/scale")),
                    name="qu")
            self.qv = \
                ed.models.Normal(
                    loc=tf.Variable(tf.random_normal([self.P, self.K]),
                                    name="qv/loc"),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.P, self.K]), name="qv/scale")),
                    name="qv")
            self.qe = \
                ed.models.Normal(
                    loc=tf.Variable(
                        tf.random_normal([self.N, self.P]), name="qe/loc"),
                    scale=tf.nn.softplus(
                        tf.Variable(tf.random_normal([self.N, self.P]),
                                    name="qe/scale")),
                    name="qe")
        self.inference = \
            ed.KLqp(latent_vars={self.u: self.qu, self.v: self.qv, self.e: self.qe},
                    data={self.y: self.data})

        self.inference.initialize(n_iter=n_iter, n_samples=n_samples)


class hmc(algorithm):
    def __init__(self, y, u, v, e, data, K):
        self.y = y
        self.u = u
        self.v = v
        self.e = e
        self.data = data
        self.N = data.shape[0]
        self.P = data.shape[1]
        self.K = K

    def initialize(self, n_iter=500, step_size=0.25, n_steps=10):
        with tf.name_scope("posterior_hmc"):
            self.qu = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, self.N, self.K]), name="qu"))
            self.qv = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, self.P, self.K]), name="qu"))
            self.qe = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, self.N, self.P]), name="qe"))

        self.inference = \
            ed.HMC(latent_vars={self.u: self.qu, self.v: self.qv, self.e: self.qe},
                   data={self.y: self.data})

        self.inference.initialize(step_size=step_size, n_steps=n_steps)


class mh(algorithm):
    def __init__(self, y, u, v, e, data, K):
        self.y = y
        self.u = u
        self.v = v
        self.e = e
        self.data = data
        self.N = data.shape[0]
        self.P = data.shape[1]
        self.K = K

    def initialize(self, n_iter = 500, step_size = 0.1):
        with tf.name_scope("posterior_mh"):
            self.qu = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, self.N, self.K]), name="qu"))
            self.qv = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, self.P, self.K]), name="qu"))
            self.qe = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, self.N, self.P]), name="qe"))

        with tf.name_scope("proposal_mh"):
            self.pu = ed.models.Normal(loc=self.u, scale=step_size * tf.ones([self.N, self.K]))
            self.pv = ed.models.Normal(loc=self.v, scale=step_size * tf.ones([self.P, self.K]))
            self.pe = ed.models.Normal(loc=self.e, scale=step_size * tf.ones([self.N, self.P]))

        self.inference = \
            ed.MetropolisHastings(
                latent_vars={self.u: self.qu, self.v: self.qv, self.e: self.qe},
                proposal_vars={self.u: self.pu, self.v: self.pv, self.e: self.pe},
                data={self.y: self.data})

        self.inference.initialize()

