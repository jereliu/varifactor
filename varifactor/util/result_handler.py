import numpy as np


def get_sample_all(result):
    """
    collect variable of interest from each chain
    :param result: result from inference of class MultiTrace
    :param varname:
    :return: dictionary of sampled parameters
    """
    container = {}

    for name in result.varnames:
        container[name] = get_sample(result, varname=name)

    return container


def get_sample(result, varname="U"):
    """
    collect variable of interest from each chain
    :param result: result from inference of class MultiTrace
    :param varname: target variable name
    :return: np.array of dimension chain_id x iteration x matrix dimension
    """
    stat = result.get_values(varname, combine=False)
    stat = np.asarray(stat)  # chain_id x iteration x matrix dimension
    return stat
