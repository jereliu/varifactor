from collections import namedtuple

param_model = {
    'y': {'family': 'Gaussian'},
    'theta': {'k': 5, 'eps_sd': 0.1},
    'u': {'var': 0.1, 'transform': 'identity', 'cov': None},
    'v': {'var': 0.1, 'transform': 'identity'}
    }

param_model = namedtuple('param', param_model.keys())(**param_model)
