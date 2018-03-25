from __future__ import division

import datetime, time

import numpy as np

import logging
import pymc3 as pm

from varifactor.model import NEFactorModel

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")


class NEFactorInference:
    def __init__(self, model, param):
        """
        initialize inference module
        :param model: model specified by varifactor.model.NEFactorModel
        :param param: inference parameters specified in varifactor.util.setting
        """
        # check model type is pymc3
        if not isinstance(model, NEFactorModel):
            err_msg = "'model' type should be NEFactorModel, got: "
            raise ValueError(err_msg + type(model).__name__)

        logging.info('\n====================')
        logging.info('initializing inference algorithms')

        self.model = model.model
        self.chains = int(param.chains)
        self.tune = int(param.tune)
        self.method = param.method
        self.setting = param.setting
        self.start = param.start
        self._run_sampler = \
            {
                'Metropolis': self.run_metro,
                'Slice': self.run_slice,
                'NUTS': self.run_nuts,
                'ADVI': self.run_advi,
                'NFVI': self.run_nfvi,
                'SVGD': self.run_svgd,
            }

        if isinstance(self.start, str):
            if self.start is 'None':
                logging.info("setting param.start to None..")
                self.start = None
            elif self.start is 'MAP':
                logging.info("setting param.start to MAP..")
                self.start = pm.find_MAP(model=self.model)
            elif self.start is 'zero':
                self.start = dict()
                for var in self.model.unobserved_RVs:
                    if var.name in ["U", "V", "e"]:
                        logging.info("setting param.start for %s to 0.." % var.name)
                        self.start[var.name] = var.init_value * 0
            else:
                raise ValueError("unrecognized initialization method: self.start=" + self.start)
        else:
            logging.info("starting value is hand-specified by user. ")

        logging.info('initialization done')
        logging.info('to perform inference with a method, execute .run(method=method)')
        logging.info('\n====================')

    def run(self, method=None):
        """
            wrapper function that runs algorithm specified by self.method

        :param method: method of choice, if none then use default value self.method
        :return:
        """

        if method is None:
            method_name = self.method
        else:
            method_name = method

        result = self._run_sampler[method_name]()

        return result

    def run_metro(self, setting=None):
        """
            Metropolis-Hasting

        :param setting:
        :return: class:pymc3.backends.base.MultiTrace
        """

        logging.info('\n====================')
        logging.info('running Metropolis-Hasting..')

        # prepare setting
        if setting is None:
            setting = self.setting['Metropolis'].copy()
            n_iter = setting['n_iter']

            setting.pop('n_iter')

        # sampling
        with self.model:
            mc = pm.Metropolis(**setting)

            t0 = time.time()
            result = pm.sample(step=mc,
                               draws=n_iter,
                               chains=self.chains,
                               random_seed=_random_seed(),
                               start=self.start, tune=self.tune,
                               discard_tuned_samples=False)
            t1 = time.time()

        logging.info('Done!')
        logging.info('\n====================')

        # aftermath: add signature & record time (seconds)
        result.method_type = "mc"
        result.iter_time = (t1 - t0)/(n_iter + self.tune)

        return result

    def run_slice(self, setting=None):
        """
            Elliptical Slice Sampling

        Use slice sampler for U, V and NUTS (with default setting) for the rest

        :param setting:
        :return: class:pymc3.backends.base.MultiTrace
        """

        logging.info('\n====================')
        logging.info('running Elliptical Slice Sampling..')

        # TODO: adaptive covariance structure for prior
        if setting is None:
            setting = self.setting['Slice'].copy()
            n_iter = setting['n_iter']

            setting.pop('n_iter')

        with self.model as fa_model:
            # configure covariance matrix
            # TODO: allow non-identity covariance structure
            logging.warning('Setting Prior Covariance to Identity..')
            # u_var = fa_model["U"].distribution.variance.eval()
            # v_var = fa_model["V"].distribution.variance.eval()
            # u_diag = np.ones(fa_model["U"].init_value.size) * u_var
            # v_diag = np.ones(fa_model["V"].init_value.size) * v_var
            # prior_cov = np.diag(np.concatenate((u_diag, v_diag)))
            prior_cov = \
                np.diag(
                    np.ones(fa_model["U"].init_value.size +
                            fa_model["V"].init_value.size)
                )

            # setup sampler
            var_set_uv = [fa_model["U"], fa_model["V"]]
            mc_uv = pm.EllipticalSlice(var_set_uv, prior_cov=prior_cov)  # no hyper-parameters
            mc_set = [mc_uv]

            if len(fa_model.unobserved_RVs) > 2:
                # if there are other parameters, sample those using NUTS
                var_set_other = \
                    [RV for RV in fa_model.unobserved_RVs if RV.name not in ["U", "V"]]
                mc_other = pm.NUTS(var_set_other)
                mc_set = [mc_uv, mc_other]

            # sampling
            t0 = time.time()
            result = pm.sample(step=mc_set,
                               draws=n_iter,
                               chains=self.chains,
                               random_seed=_random_seed(),
                               start=self.start, tune=self.tune,
                               discard_tuned_samples=False)
            t1 = time.time()

        logging.info('Done!')
        logging.info('\n====================')

        # aftermath: add signature & record time (seconds)
        result.method_type = "mc"
        result.iter_time = (t1 - t0)/(n_iter + self.tune)

        return result

    def run_nuts(self, setting=None):
        """
            No-U-Turn Sampler

        :param setting:
        :return: class:pymc3.backends.base.MultiTrace
        """
        logging.info('\n====================')
        logging.info('running NUTS..')

        # prepare setting
        if setting is None:
            setting = self.setting['NUTS'].copy()
            n_iter = setting['n_iter']

            setting.pop('n_iter')

        # sampling
        with self.model:
            mc = pm.NUTS(**setting)

            t0 = time.time()
            result = pm.sample(step=mc,
                               draws=n_iter,
                               chains=self.chains,
                               random_seed=_random_seed(),
                               start=self.start, tune=self.tune,
                               discard_tuned_samples=False)
            t1 = time.time()

        logging.info('Done!')
        logging.info('\n====================')

        # aftermath: add signature & record time (seconds)
        result.method_type = "mc"
        result.iter_time = (t1 - t0)/(n_iter + self.tune)

        return result

    def run_advi(self, setting=None, track=True):
        """
            Mean-field Variational Inference

        :param setting:
        :param track: whether collect algorithm samples *during* optimization (i.e. add _single_sample to callback)
        :return: class:pymc3.Approximation
        """

        logging.info('\n====================')
        logging.info('running Mean Field VI..')

        # prepare setting
        if setting is None:
            # for compatible purpose, ADVI doesn't have parameters
            setting = self.setting['ADVI'].copy()
            vi_freq = setting['vi_freq']
            sample_freq = setting['sample_freq']

            setting.pop('vi_freq')
            setting.pop('sample_freq')

        # sampling
        with self.model:
            vi = pm.ADVI(start=self.start,
                         random_seed=_random_seed())

            if track:
                tracker = [
                    pm.callbacks.Tracker(
                        sample=_single_sample(sample_freq=sample_freq),)
                ]
            else:
                tracker = None

            t0 = time.time()
            result = vi.fit(n=self.tune * vi_freq,
                            obj_optimizer=pm.adam(learning_rate=.001),
                            callbacks=tracker)
            t1 = time.time()

        logging.info('Done!')
        logging.info('\n====================')

        # extract result
        if track:
            tracker = tracker[0]
            result.sample_tracker = tracker['sample']
        else:
            result.sample_tracker = \
                result.sample(draws=int(self.tune * vi_freq/100))

        # aftermath: add signature & record time (seconds)
        result.method_type = "vi"
        result.iter_time = (t1 - t0)/(self.tune * vi_freq)

        return result

    def run_nfvi(self, setting=None, track=True):
        """
            Variational Inference with Normalizing Flow

        :param setting:
        :param track: whether collect algorithm samples *during* optimization (i.e. add _single_sample to callback)
        :return: class:pymc3.Approximation
        """

        logging.info('\n====================')
        logging.info('running Normalizing Flow VI..')

        # prepare setting
        if setting is None:
            setting = self.setting['NFVI'].copy()
            vi_freq = setting['vi_freq']
            sample_freq = setting['sample_freq']

            setting.pop('vi_freq')
            setting.pop('sample_freq')

        # sampling
        with self.model:
            vi = pm.NFVI(start=self.start,
                         random_seed=_random_seed(),
                         **setting)

            if track:
                tracker = [
                    pm.callbacks.Tracker(
                        sample=_single_sample(sample_freq=sample_freq),)
                ]
            else:
                tracker = None

            t0 = time.time()
            result = vi.fit(n=self.tune * vi_freq,
                            obj_optimizer=pm.adam(learning_rate=.001),
                            callbacks=tracker)
            t1 = time.time()

        logging.info('Done!')
        logging.info('\n====================')

        # extract result
        if track:
            tracker = tracker[0]
            result.sample_tracker = tracker['sample']
        else:
            result.sample_tracker = \
                result.sample(draws=int(self.tune * vi_freq/100))

        # aftermath: add signature & record time (seconds)
        result.method_type = "vi"
        result.iter_time = (t1 - t0)/(self.tune * vi_freq)

        return result

    def run_svgd(self, setting=None, track=True):
        """
            Stein Variational Gradient Descent

        :param setting:
        :param track: whether collect algorithm samples *during* optimization (i.e. add _single_sample to callback)
        :return: class:pymc3.Approximation
        """

        logging.info('\n====================')
        logging.info('running Stein Variational GD..')

        # prepare setting
        if setting is None:
            setting = self.setting['SVGD'].copy()

            vi_freq = setting['vi_freq']
            sample_freq = setting['sample_freq']

            setting.pop('vi_freq')
            setting.pop('sample_freq')

        # sampling
        with self.model:
            vi = pm.SVGD(start=self.start,
                         random_seed=_random_seed(),
                         **setting)

            if track:
                tracker = [
                    pm.callbacks.Tracker(
                        sample=_single_sample(sample_freq=sample_freq),)
                ]
            else:
                tracker = None

            t0 = time.time()
            result = vi.fit(n=self.tune * vi_freq, callbacks=tracker)
            t1 = time.time()

        logging.info('Done!')
        logging.info('\n====================')

        # extract result
        if track:
            tracker = tracker[0]
            result.sample_tracker = tracker['sample']
        else:
            result.sample_tracker = \
                result.sample(draws=int(self.tune * vi_freq/100))

        # aftermath: add signature & record time (seconds)
        result.method_type = "vi"
        result.iter_time = (t1 - t0)/(self.tune * vi_freq)

        return result


def _single_sample(sample_freq=100):
    """
    callback function used to sample from VI iterations
    format follows specification in pymc3.variational.inference.py/_iterate_with_loss

    :param approx:
    :param _: score
    :param iter:
    :param sample_freq: frequency to sample
    :return: a sample (one draw) from target distribution
    """
    def sample_func(approx, _, iter):
        if iter % int(sample_freq) == 0:
            return approx.sample(draws=1)
        else:
            pass

    return sample_func


def _random_seed():
    t = datetime.datetime.now()
    return int(time.mktime(t.timetuple()))
