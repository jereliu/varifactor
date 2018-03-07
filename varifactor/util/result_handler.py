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
    # get result
    if result.method_type == "vi":
        stat = [x[varname] for x in result.sample_tracker if x is not None]
        stat = np.moveaxis(stat, 0, 1)
        # if only one chain, then remove the chain dimension
        if stat.shape[0] == 1:
            stat = stat[0]

    elif result.method_type == "mc":
        stat = result.get_values(varname, combine=False)

    # convert to array
    stat = np.asarray(stat)  # chain_id x iteration x matrix dimension

    # adjust matrix dimension
    if varname in ["U", "V"]:
        stat = np.moveaxis(stat, -2, -1)

    return stat
