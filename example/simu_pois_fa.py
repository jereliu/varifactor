# Poisson factor analysis using pyMC3

import numpy as np

from varifactor import simu
from varifactor.model import NEFactorModel as Model
from varifactor.inference import NEFactorInference as Infer
from varifactor.metric.moment import MomentDistance
from varifactor.metric.kernel import KSD

from varifactor.setting import param_model, param_infer

from varifactor.util.result_handler import get_sample


#########################
# 1. Set Parameters  ####
#########################

family = ["Gaussian", "Binomial", "Poisson"][2]

N = 100
P = 10
K = 2


#########################
# 2. Run Simulation  ####
#########################

n_chain = 500

for method in ['ADVI', 'Metropolis', 'NUTS']:
    if method is 'ADVI':
        itersize = (param_infer.n * param_infer.vi_freq)/100
    else:
        itersize = param_infer.n + param_infer.tune

    U_sample_all = np.zeros((n_chain, itersize, N, param_model.theta['k']))
    V_sample_all = np.zeros((n_chain, itersize, P, param_model.theta['k']))
    
    total_iter = n_chain/param_infer.chains
    for iter_num in range(total_iter):
        print("##################################")
        print("######## Iteration %d/%d #########" % (iter_num, total_iter))
        print("##################################")
        # generate data
        y_train, u_train, v_train, e_train = \
            simu.data(N, P, K, family=family, eps_sd=0,
                      uv_scale=[param_model.u['sd'], param_model.v['sd']])

        # initialize model
        nefm_model = Model(y_train, param_model, e=e_train)

        # initialize inference
        nefm_infer = Infer(nefm_model, param_infer)

        # run algorithm
        result = nefm_infer.run(method=method)

        # store result
        idx_range = range(
            (iter_num*param_infer.chains), (iter_num*param_infer.chains + param_infer.chains))

        U_sample_all[idx_range, :, :, :] = get_sample(result, "U")
        np.save('./result/%s_%s_U' % (method, family), U_sample_all)

        V_sample_all[idx_range, :, :, :] = get_sample(result, "V")
        np.save('./result/%s_%s_V' % (method, family), V_sample_all)
