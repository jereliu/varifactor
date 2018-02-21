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
        :param param: inference parameters specified in varifactor.setting
        """
        # check model type is pymc3
        if not isinstance(model, NEFactorModel):
            err_msg = "'model' type should be NEFactorModel, got: "
            raise ValueError(err_msg + type(model).__name__)

        logging.info('\n====================')
        logging.info('initializing inference algorithms')

        self.model = model.model
        self.n = int(param.n)
        self.chains = int(param.chains)
        self.tune = int(param.tune)
        self.vi_freq = int(param.vi_freq)
        self.method = param.method
        self.setting = param.setting
        self.start = param.start
        self._run_sampler = \
            {
                'Metropolis': self.run_metro,
                'NUTS': self.run_nuts,
                'ADVI': self.run_advi,
                'NFVI': self.run_nfvi,
                'OPVI': self.run_opvi,
                'SVGD': self.run_svgd,
            }

        if self.start is None:
            logging.info("no starting value provided! (i.e. param.start = None)")
            logging.info("setting param.start to posterior mode..")
            self.start = pm.find_MAP(model=self.model)
            logging.info("setting param.start to zero..")
            for key in self.start.keys():
                self.start[key] = self.start[key] * 0


        logging.info('initialization done')
        logging.info('to perform inference, execute .run()')
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

        method_setting = self.setting[method_name]
        result = self._run_sampler[method_name](setting=method_setting)

        return result

    def run_metro(self, setting=None):
        logging.info('\n====================')
        logging.info('running Metropolis-Hasting..')

        if setting is None:
            setting = self.setting['Metropolis']

        with self.model:
            mc = pm.Metropolis(**setting)
            result = pm.sample(step=mc,
                               draws=self.n,
                               chains=self.chains,
                               start=self.start, tune=self.tune,
                               discard_tuned_samples=False)

        logging.info('Done!')
        logging.info('\n====================')

        result.method_type = "mc"

        return result

    def run_nuts(self, setting=None):
        logging.info('\n====================')
        logging.info('running NUTS..')

        if setting is None:
            setting = self.setting['NUTS']

        with self.model:
            mc = pm.NUTS(**setting)
            result = pm.sample(step=mc,
                               draws=self.n,
                               chains=self.chains,
                               start=self.start, tune=self.tune,
                               discard_tuned_samples=False)

        logging.info('Done!')
        logging.info('\n====================')

        result.method_type = "mc"

        return result

    def run_advi(self, setting=None):
        logging.info('\n====================')
        logging.info('running Mean Field VI..')

        if setting is None:
            # for compatible purpose, ADVI doesn't have parameters
            setting = self.setting['ADVI']

        # setup vi and tracker
        with self.model:
            vi = pm.ADVI(start=self.start)

            tracker = pm.callbacks.Tracker(
                sample=_single_sample,
            )

            result = vi.fit(n=self.n * self.vi_freq, callbacks=[tracker])

        logging.info('Done!')
        logging.info('\n====================')

        result.sample_tracker = tracker['sample']
        result.method_type = "vi"

        return result

    def run_nfvi(self, setting=None):
        logging.info('\n====================')
        logging.info('running Norm Flow VI..')

        if setting is None:
            setting = self.setting['NFVI']

        with self.model:
            vi = pm.NFVI(start=self.start, **setting)

            tracker = pm.callbacks.Tracker(
                sample=_single_sample
            )

            result = vi.fit(n=self.n * self.vi_freq, callbacks=[tracker])

        logging.info('Done!')
        logging.info('\n====================')

        result.sample_tracker = tracker['sample']
        result.method_type = "vi"

        return result

    def run_opvi(self, setting=None):
        raise NotImplementedError

    def run_svgd(self, setting=None):
        raise NotImplementedError



def _single_sample(approx, _, iter):
    """
    callback function used to sample from VI iterations
    :param approx:
    :param _:
    :param iter:
    :return: a sample (one draw) from target distribution
    """
    if iter % 100 == 0:
        return approx.sample(draws=1)
    else:
        pass


