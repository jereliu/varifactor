# Poisson factor analysis using pyMC3
import os
os.environ['MKL_THREADING_LAYER'] = "GNU"

import numpy as np
import scipy.stats as st
import theano as tt

from varifactor.model import NEFactorModel as Model
from varifactor.inference import NEFactorInference as Infer
from varifactor.util.setting import param_model, param_infer

from varifactor.util import simu
from varifactor.util.result_handler import get_sample

import matplotlib.pyplot as plt
import seaborn as sns


report_addr = \
    "/home/jeremiah/Dropbox/Research/Harvard/Thesis/Lorenzo/" \
    "1. varifactor/Report/Progress/2018_03_Week_2/plot/"


#################################################
# 0. helper function                         ####
#################################################

def contour_2d(data):
    """

    :param data: a Nx2 npy tensor
    :return:
    """
    x = data[:, 0]
    y = data[:, 1]

    # define grid
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # if no function, do kernel density estimation based on data
    values = np.vstack([x, y])
    f = st.gaussian_kde(values)
    z = np.reshape(f(positions).T, xx.shape)

    plt.contourf(xx, yy, z, cmap='Blues')


def contour2d_grid(sample, save_addr=None, figsize=(20, 20)):
    """

    :param sample: a N x dim1 x dim2 npy array
    :param save_addr:
    :param figsize:
    :return:
    """
    N, dim1, dim2 = sample.shape
    dim = dim1 * dim2

    # create directory if not exist
    if save_addr is not None:
        if not os.path.isdir(os.path.dirname(save_addr)):
            os.mkdir(os.path.dirname(save_addr))
        plt.ioff()

    # start ploting
    sample_plot = sample.reshape((N, dim))
    plot_id = 0
    for i in range(dim-1):
        for j in range(1, dim):
            plot_id += 1
            plt.subplot(dim-1, dim-1, plot_id)

            if i >= j:
                plt.plot()
                plt.axis('off')
            else:
                print("plotting (%d, %d)" % (i + 1, j + 1))
                contour_2d((sample_plot[:, (i, j)]))
                plt.title("(%d, %d)" % (i + 1, j + 1))

    # optionally, save
    if save_addr is not None:
        fig = plt.gcf()
        fig.set_size_inches(figsize[0], figsize[1])
        fig.savefig(save_addr, bbox_inches='tight')
        plt.close()
        plt.ion()




#################################################
# 1. Sample 2d factors from Poisson FA model ####
#################################################

family = param_model.y["family"]

N = 100
P = 3
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
method_list = ["Metropolis", "Slice", "NUTS", "ADVI", "NFVI", "SVGD"]
method_name = method_list[-1]
track_vi_during_opt = False

# for method_name in method_list[2]:
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
else:
    raise ValueError("method '%s' not supported" % method_name)

#################################
# 2. Visualize Posterior     ####
#################################
# plot
if sample.method_type == "vi" and not track_vi_during_opt:
    V_sample = get_sample(sample, "V").T
else:
    V_sample = get_sample(sample, "V")

contour2d_grid(V_sample,
               "%s/contour/%s/%s.pdf" % (report_addr, family, method_name))


# plot factor norm verses density

N, dim1, dim2 = V_sample.shape
V_sample_select = V_sample.reshape((N, dim1 * dim2))
values = V_sample_select[90000:, :3].T
kde = st.gaussian_kde(values)
density = kde(values)


plt.plot(np.sum(values**2, 0), density, 'o')

# plot 3D V density, if K==3
if K == 3:
    from mpl_toolkits.mplot3d import Axes3D
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    x, y, z = values
    ax.scatter(x, y, z, c=density)
    plt.show()


