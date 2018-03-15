from __future__ import division

import logging

import numpy as np
from numpy.linalg import inv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metric")

# TODO: inhere from master class: Metric

class MomentMetric:
    def __init__(self, model, chain_thres=50):
        """
        Computes distance to true mean and variance for Gaussian NEF model with
        V known and fixed and e = 0
        :param model:
        :param chain_thres: if number of chain is smaller than threshold, do moving avg
        """
        # check value to make sure
        #   1. family is Gaussian
        #   2. V is observed
        #   3. e is observed and fixed at 0

        name_obsvar = [var.name for var in model.model.observed_RVs]

        if not model.family == "Gaussian":
            raise ValueError("Distribution family not Gaussian")
        elif not ('y_gaussian' in name_obsvar):
            raise ValueError("outcome y not found in list of observed random variables")
        elif not ('V' in name_obsvar):
            raise ValueError("V not fixed random variable")
        elif not ('e' in name_obsvar):
            raise ValueError("e not fixed random variable")
        elif not (model.model['e'].init_value == 0).all():
            raise ValueError("e is fixed random variable but value not all equal to zero")

        self.model = model.model
        self.chain_thres = chain_thres

        # initialize variables
        v = self.model['V'].init_value.T
        y = self.model['y_gaussian'].init_value
        sigma_u2 = self.model['U'].distribution.cov[0, 0].get_scalar_constant_value()

        logging.info("sigma_u^2 was set to " + str(sigma_u2) + ". Change if needed")

        # calculate true mean and variance
        p, k = v.shape
        self.cov_true = np.linalg.inv(v.T.dot(v) + np.eye(k)/sigma_u2)
        self.mean_true = y.dot(v).dot(self.cov_true)

    def mean_dist(self, u):
        mean_list = self.mean(u)
        diff_list = (mean_list - self.mean_true.T)
        norm_list = np.sqrt(np.mean(diff_list**2, axis=(1, 2)))
        return norm_list

    def cov_dist(self, u):
        cov_list = self.cov(u)
        diff_list = (cov_list - np.linalg.inv(self.cov_true))
        norm_list = np.sqrt(np.mean(diff_list**2, axis=(1, 2)))
        return norm_list

    def mean(self, u):
        """
        :param U: a matrix of dimension num_chain x num_iter x N x K
        :return:
        """
        num_chain, num_iter, _, _ = u.shape

        # compute sample mean by iteration
        mean_list = np.mean(u, axis=0)

        # if too few chains, then do moving-window average
        if num_chain < self.chain_thres:
            mean_list = self.moving_avg(mean_list)

        return mean_list

    def cov(self, u):
        num_chain, num_iter, k, _ = u.shape

        # compute sample mean by iteration
        mean_list = np.mean(u, axis=0)
        u_center = u - mean_list

        cov_list = np.empty(shape=(num_chain, num_iter, k, k))
        for i in range(num_chain):
            for j in range(num_iter):
                if (u_center[i, j, :, :] == 0).all():
                    # skip if all entries are zero
                    continue
                else:
                    cov_list[i, j, :, :] = inv(np.cov(u_center[i, j, :, :]))
        cov_list = np.mean(cov_list, axis=0)

        # if too few chains, then do moving-window average
        if num_chain < self.chain_thres:
            cov_list = self.moving_avg(cov_list)

        return cov_list

    def moving_avg(self, stat_list, n_sample=100):
        num_iter = stat_list.shape[0]
        mv_avg = np.zeros(shape=stat_list.shape)

        for i in range(num_iter):
            lo = np.max([0, i - n_sample / 2]).astype('int')
            hi = np.min([lo + n_sample, num_iter]).astype('int')
            idx_list = range(lo, hi)
            mv_avg[i, :, :] = np.mean(stat_list[idx_list, :, :], axis=0)
        return mv_avg
