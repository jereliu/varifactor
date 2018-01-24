# Poisson factor analysis using pyMC3

import numpy as np
import pymc3 as pm
import theano.tensor as tensor

from varifactor.simu import data as simu_data
from varifactor.model_pm import NEFactorModel as Model
from varifactor.setting import param_model



# 1. Data generation ####

family = "Poisson"

N = 100
P = 15
K = 5

# generate data
y_train, u_train, v_train, e_train = simu_data(N, P, K, family=family)

nef_factor = Model(y_train, param_model)



# initialization
nef_factor = pm.Model()

with nef_factor:
    uv_trans = ["identity", "identity"]
    uv_scale = [0.1, 0.1]
    eps_sd = 0.1

    cov_u = np.eye(N)
    cov_v = np.eye(P)

    # initialize random variables
    u = pm.MvNormal(name="U", mu=0, cov=cov_u, shape=(K, N),
                    testval=np.random.normal(0, uv_scale[0], size=(K, N)))
    v = pm.MvNormal(name="V", mu=0, cov=cov_v, shape=(K, P),
                    testval=np.random.normal(0, uv_scale[1], size=(K, P)))
    u = u.T
    v = v.T

    e = pm.Normal(name="e", mu=0, sd=1, shape=(N, P),
                  testval=np.random.normal(0, eps_sd, size=(N, P)))

    # factor transformation
    u_t = pm.Deterministic("Ut", _factor_transform(u, name=uv_trans[0]))
    v_t = pm.Deterministic("Vt", _factor_transform(v, name=uv_trans[1]))

    # theta
    theta = pm.Deterministic("theta", tensor.dot(u, v.T) + e)
    y = _nef_family(name="y", theta=theta, n=N, p=P,
                    family=family, observed=y_train)



#### 3. Sampling ####
#### 3.1 MCMC ====

# Metropolis-Hasting
with nef_factor.model:
    step_mh = pm.Metropolis()
    sample_mh = pm.sample(step=step_mh)

# NUTS
with nef_factor:
    step_hmc = pm.NUTS()
    sample_hmc = pm.sample(step=step_hmc)

#### 3.2 VIs ====

# KL-MF
with nef_factor:
    fit_klmf = pm.ADVI()
    sample_klmf = fit_klmf.sample(500)

# KL-MF-flow
with nef_factor:
    fit_klmf_flow = pm.NFVI('planar*8')
    sample_klmf_flow = fit_klmf_flow.approx.sample(500)


# Stein
with nef_factor:
    fit_svgd = pm.SVGD(n_particles=100, jitter=1)
    sample_svgd = fit_svgd.approx.sample(500)
