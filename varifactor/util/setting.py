from collections import namedtuple

param_model = {
    'y': {'family': 'Poisson'},
    'theta': {'k': 2, 'eps_sd': 0},
    'u': {'sd': 0.2, 'transform': 'identity'},
    'v': {'sd': 2, 'transform': 'identity', 'cov': None}
    }


param_infer = {
    'chains': 1,
    'tune': 1000,
    'method': 'NUTS',
    'start': 'zero', # str (MAP / zero /None) indicating init method, or a dict of actual init values
    'setting': {
        'Metropolis': {'n_iter': 1000,
                       'scaling': 100.},
        'Slice': {'n_iter': 1000,
                  'scaling': 100.},
        'NUTS': {'n_iter': 1000,
                 'target_accept': 0.8, 'max_treedepth': 10},
        'ADVI': {'vi_freq': 200, 'sample_freq': 100},
        'NFVI': {'vi_freq': 50, 'sample_freq': 25,
                 'flow': 'scale-hh*20-loc-radial*2', 'jitter': 1.},
        'SVGD': {'vi_freq': 20, 'sample_freq': 10,
                 'n_particles': 100, 'jitter': 1.}
    }
}

param_model = namedtuple('model_param', param_model.keys())(**param_model)
param_infer = namedtuple('infer_param', param_infer.keys())(**param_infer)


