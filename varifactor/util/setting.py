from collections import namedtuple

param_model = {
    'y': {'family': 'Gaussian'},
    'theta': {'k': 2, 'eps_sd': 0},
    'u': {'sd': 0.2, 'transform': 'identity'},
    'v': {'sd': 2, 'transform': 'identity', 'cov': None}
    }


param_infer = {
    'chains': 1,
    'n': 1000,
    'tune': 1000,
    'method': 'NUTS',
    'start': 'zero', # str (MAP / zero) indicating init method, or a dict of actual init values
    'setting': {
        'Metropolis': {'scaling': 100.},
        'Slice': {'scaling': 100.},
        'NUTS': {'target_accept': 0.8, 'max_treedepth': 10},
        'ADVI': {'vi_freq': 200, 'sample_freq': 100},
        'NFVI': {'flow': 'planar*8', 'jitter': 1., 'vi_freq': 50, 'sample_freq': 25},
        'SVGD': {'n_particles': 100, 'jitter': 1., 'vi_freq': 20, 'sample_freq': 10}
    }
}

param_model = namedtuple('model_param', param_model.keys())(**param_model)
param_infer = namedtuple('infer_param', param_infer.keys())(**param_infer)


