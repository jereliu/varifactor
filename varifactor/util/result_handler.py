import os

from tqdm import tqdm

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


def get_npy(addr, ext="npy"):
    """
    assume address contains npy files with name "%d.npy", %d being an integer
    assume all file are of the same dimension
    :param addr:
    :return:
    """

    fname_lst = [fname for fname in os.listdir(addr) if "." + ext in fname]

    if ext == "npy":
        # attempt reading the first file
        try:
            sample = np.load(addr + fname_lst[0])
        except IOError:
            print("cannot read npy file " + str(fname_lst[0]))

        # extract dimension and build container
        N, P, K = sample.shape
        sample_list = np.zeros(shape=[len(fname_lst), N, P, K])

        # file reading in
        for i in tqdm(range(len(fname_lst))):
            sample_list[i] = np.load(addr + str(i) + ".npy")

    else:
        raise ValueError("Unsupported extension %s" % (ext))

    return sample_list
