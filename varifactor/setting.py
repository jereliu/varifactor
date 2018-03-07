from collections import namedtuple

param_model = {
    'y': {'family': 'Poisson'},
    'theta': {'k': 2, 'eps_sd': 0.1},
    'u': {'sd': 0.2, 'transform': 'identity'},
    'v': {'sd': 2, 'transform': 'identity', 'cov': None}
    }


param_infer = {
    'chains': 1,
    'n': 1000,
    'tune': 1000,
    'vi_freq': 20,   # how frequent to sample from VI iterations
    'method': 'NUTS',
    'start': 'zero', # str (MAP / zero) indicating init method, or a dict of actual init values
    'setting': {
        'Metropolis': {'scaling': 100.},
        'NUTS': {'target_accept': 0.8, 'max_treedepth': 10},
        'ADVI': {'vi_freq': 1000},
        'NFVI': {'flow': 'planar*10', 'jitter': 0.1, 'vi_freq': 1000},
        'SVGD': {},
        'OPVI': {}
    }
}

param_model = namedtuple('model_param', param_model.keys())(**param_model)
param_infer = namedtuple('infer_param', param_infer.keys())(**param_infer)


