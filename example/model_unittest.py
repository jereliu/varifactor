# Poisson factor analysis using pyMC3

import numpy as np
import theano as tt

from varifactor.model import NEFactorModel as Model
from varifactor.inference import NEFactorInference as Infer
from varifactor.metric.moment import MomentMetric
from varifactor.metric.kernel import KSD

from varifactor.util.setting import param_model, param_infer

from varifactor.util import simu
from varifactor.util.result_handler import get_sample


#########################
# 1. Data generation ####
#########################

family = ["Gaussian", "Binomial", "Poisson"][2]

N = 50
P = 5
K = 2

# generate data
y_train, u_train, v_train, e_train = \
    simu.data(N, P, K, family=family, eps_sd=0,
              uv_scale=[param_model.u['sd'], param_model.v['sd']])

y_train = tt.shared(y_train)

#########################
# 2. Model Setup     ####
#########################

# initialize model
nefm_model = Model(y_train, param_model, e=e_train)

# initialize inference
nefm_infer = Infer(nefm_model, param_infer)

#########################
# 3. Run Inference   ####
#########################

# run algorithm
sample = dict()
for method in ['Metropolis', 'NUTS', 'ADVI']:
    result = nefm_infer.run(method=method)
    sample[method] = get_sample(result, "U")

#########################
# 4. Evaluation      ####
#########################

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
for method in ['Metropolis', 'NUTS', 'ADVI']:
    plt.plot(sample[method][:, 0, 0])
plt.show()

