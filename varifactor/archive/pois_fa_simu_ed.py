import tensorflow as tf
import edward as ed
from varifactor.simu import data as simu_data
from varifactor.model_ed import nef_factor as model
import varifactor.algorithm_ed as algorithm


family = "Poisson"

N = 100
P = 15
K = 5

# prepare data
y_train, u_train, v_train, e_train = simu_data(N, P, K, family=family)


# build model
with tf.name_scope("model"):
    y_model, u_model, v_model, e_model = model(N, P, K, family=family)

# choose algorithm
infer_mh = \
    algorithm.mh(y_model, u_model, v_model, e_model, y_train, K,
                 n_iter=1000, step_size=1e-3)
infer_hmc = \
    algorithm.hmc(y_model, u_model, v_model, e_model, y_train, K,
                  n_iter=1000, step_size=1e-3, n_steps=5)

# perform inference
infer_mh.run(n_print=100, logdir='log/gibss')
infer_hmc.run(n_print=100, step_size=1e-3, logdir='log/hmc')