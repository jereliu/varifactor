# Poisson factor analysis using pyMC3
import os
os.environ['MKL_THREADING_LAYER'] = "GNU"

import numpy as np
import theano as tt

from varifactor.model import NEFactorModel as Model
from varifactor.inference import NEFactorInference as Infer
from varifactor.util.setting import param_model, param_infer

from varifactor.util import simu
from varifactor.util.result_handler import get_sample

import matplotlib.pyplot as plt
import seaborn as sns


#################################################
# 1. Sample 2d factors from Poisson FA model ####
#################################################

family = "Poisson"

N = 5
P = 2
K = 2

# generate data
y_train, u_train, v_train, e_train = \
    simu.data(N, P, K, family=family, eps_sd=0,
              uv_scale=[param_model.u['sd'], param_model.v['sd']])

y_train = tt.shared(y_train)

# initialize model
nefm_model = Model(y_train, param_model, e=e_train)

# initialize inference
nefm_infer = Infer(nefm_model, param_infer)

# run method of choice
method_name = ["Metropolis", "Slice", "NUTS", "ADVI", "NFVI", "SVGD"][1]
track_vi_during_opt = False

if method_name == "Metropolis":
    sample = nefm_infer.run_metro()
elif method_name == "Slice":
    sample = nefm_infer.run_slice()
elif method_name == "NUTS":
    sample = nefm_infer.run_nuts()
elif method_name == "ADVI":
    sample = nefm_infer.run_advi(track=track_vi_during_opt)
elif method_name == "NFVI":
    sample = nefm_infer.run_nfvi(track=track_vi_during_opt)
elif method_name == "SVGD":
    sample = nefm_infer.run_svgd(track=track_vi_during_opt)

#################################
# 2. Visualize Posterior     ####
#################################
# plot
if sample.method_type == "vi" and not track_vi_during_opt:
    V_sample = get_sample(sample, "V")[:, 0, :].T
else:
    V_sample = get_sample(sample, "V")[:, 0, :]

sns.jointplot(x=V_sample[:, 0], y=V_sample[:, 1], kind='kde',
              xlim=(-4, 4), ylim=(-4, 4))

# plot factor norm verses density
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

values = V_sample.T
kde = stats.gaussian_kde(values)
density = kde(values)


plt.plot(np.sum(values**2, 0), density, 'o')

# plot 3D V density, if K==3
if K == 3:
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    x, y, z = values
    ax.scatter(x, y, z, c=density)
    plt.show()
