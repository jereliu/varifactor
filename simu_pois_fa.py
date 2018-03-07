# Poisson factor analysis using pyMC3
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import multiprocessing as mp

import numpy as np
import theano

from varifactor import simu
from varifactor.model import NEFactorModel as Model
from varifactor.inference import NEFactorInference as Infer

from varifactor.setting import param_model, param_infer

from varifactor.util.result_handler import get_sample

# fp=open('memory_profiler.log','w+')
# @profile(stream=fp)


def run_simu(inference, n_chain=500, methods=['ADVI', 'Metropolis', 'NUTS'],
             res_addr="./result/", task_name=None):


    # set up result folder
    if task_name is None:
        task_name = "%s_n%d_p%d_k%d" % (family, N, P, K)

    res_addr = res_addr + task_name + "/"

    if not os.path.isdir(res_addr):
        os.mkdir(res_addr)

    for method_name in methods:
        # set up result folder
        if not os.path.exists(res_addr + method_name):
            os.mkdir(res_addr + method_name)
            os.mkdir(res_addr + method_name + "/U")
            os.mkdir(res_addr + method_name + "/V")

        # # set container size for algorithm iterations
        # if method_name is 'ADVI':
        #     iter_size = (param_infer.n * param_infer.vi_freq)/100
        # else:
        #     iter_size = param_infer.n + param_infer.tune

        # run sampler
        total_iter = n_chain/param_infer.chains
        for iter_num in range(total_iter):
            # U_sample_all = np.zeros((n_chain, iter_size, N, param_model.theta['k']))
            # V_sample_all = np.zeros((n_chain, iter_size, P, param_model.theta['k']))

            print("##################################")
            print("######## Iteration %d/%d #########" % (iter_num+1, total_iter))
            print("##################################")
            # generate data ----
            y_train, u_train, v_train, e_train = \
                simu.data(N, P, K, family=family, eps_sd=0,
                          uv_scale=[param_model.u['sd'], param_model.v['sd']])
            y_shared.set_value(y_train)  # update model data

            # # check if data update successful
            # print(y_train[0, :])
            # print(inference.model['y'].observations.eval()[0,:])

            # run algorithm (in multiprocessing to prevent memory leak)----
            q = mp.Queue()
            p = mp.Process(target=run_inference, args=(q, inference, method_name))
            p.start()
            result = q.get()
            q.close()
            p.join()

            # store result and initial values ----
            u_sample = \
                np.concatenate((u_train[np.newaxis, :, :], get_sample(result, "U")))
            v_sample = \
                np.concatenate((v_train[np.newaxis, :, :], get_sample(result, "V")))

            np.save(res_addr + method_name + "/U/%d" % (iter_num), u_sample)
            np.save(res_addr + method_name + "/V/%d" % (iter_num), v_sample)

    return None


def run_inference(q, inference, method_name):
    result = inference.run(method=method_name)
    q.put(result)


if __name__ == "__main__":
    res_path = os.path.dirname(os.path.abspath(__file__)) + "/result/"

    #########################
    # 1. Initialize Model  ##
    #########################

    family = ["Gaussian", "Binomial", "Poisson"][2]

    N = 50
    P = 5
    K = 2

    y_train, u_train, v_train, e_train = \
        simu.data(N, P, K, family=family, eps_sd=0,
                  uv_scale=[param_model.u['sd'], param_model.v['sd']])

    y_shared = theano.shared(y_train)
    nefm_model = Model(y_shared, param_model, e=e_train)
    nefm_infer = Infer(nefm_model, param_infer)

    #########################
    # 2. Run Simulation  ####
    #########################
    run_simu(nefm_infer, n_chain=500,
             methods=['ADVI', 'Metropolis', 'NUTS'],
             res_addr=res_path)
