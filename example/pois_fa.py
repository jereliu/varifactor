# Poisson factor analysis using pyMC3

from varifactor.simu import data as generate_data
from varifactor.model import NEFactorModel as Model
from varifactor.inference import NEFactorInference as Infer

from varifactor.setting import param_model, param_infer

# 1. Data generation ####

family = ["Gaussian", "Binomial", "Poisson"][1]

N = 100
P = 15
K = 5

# generate data
y_train, u_train, v_train, e_train = generate_data(N, P, K, family=family)

# 2. Model Setup
nefm_model = Model(y_train, param_model)
nefm_infer = Infer(nefm_model, param_infer)

result_mh = nefm_infer.run_metro()
result_nu = nefm_infer.run_nuts()
result_ad = nefm_infer.run_advi()
result_nf = nefm_infer.run_nfvi()

result = nefm_infer.run()

