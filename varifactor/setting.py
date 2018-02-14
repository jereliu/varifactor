from collections import namedtuple

param_model = {
    'y': {'family': 'Gaussian'},
    'theta': {'k': 5, 'eps_sd': 0.1},
    'u': {'var': 0.1, 'transform': 'identity'},
    'v': {'var': 0.1, 'transform': 'identity', 'cov': None}
    }


param_infer = {
    'n': 1e4,
    'chains': 4,
    'method': 'NUTS',
    'start': None,
    'setting': {
        'Metropolis': {'scaling': 1.},
        'NUTS': {'target_accept': 0.8, 'max_treedepth': 10},
        'ADVI': {}, # advi does not have non-trivial parameters...
        'NFVI': {'flow': 'planar*10', 'jitter': 0.1},
        'SVGD': {},
        'OPVI': {}
    }
}

param_model = namedtuple('model_param', param_model.keys())(**param_model)
param_infer = namedtuple('infer_param', param_infer.keys())(**param_infer)


