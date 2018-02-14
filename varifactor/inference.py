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
        sample = self._run_sampler[method_name](setting=method_setting)

        return sample

    def run_metro(self, setting=None):
        logging.info('\n====================')
        logging.info('running Metropolis-Hasting..')

        if setting is None:
            setting = self.setting['Metropolis']

        with self.model:
            mc = pm.Metropolis(**setting)
            sample = pm.sample(step=mc,
                               draws=self.n, cores=self.chains,
                               start=self.start, tune=0)

        logging.info('Done!')
        logging.info('\n====================')

        return sample

    def run_nuts(self, setting=None):
        logging.info('\n====================')
        logging.info('running NUTS..')

        if setting is None:
            setting = self.setting['NUTS']

        with self.model:
            mc = pm.NUTS(**setting)
            sample = pm.sample(step=mc,
                               draws=self.n, cores=self.chains,
                               start=self.start, tune=0)

        logging.info('Done!')
        logging.info('\n====================')

        return sample

    def run_advi(self, setting=None):
        logging.info('\n====================')
        logging.info('running Auto Diff VI..')

        if setting is None:
            # for compatible purpose, ADVI doesn't have parameters
            setting = self.setting['ADVI']

        with self.model:
            vi = pm.ADVI(start=self.start)

            tracker = pm.callbacks.Tracker(
                mean=vi.approx.mean.eval,  # callable that returns mean
                std=vi.approx.std.eval  # callable that returns std
            )

            sample = vi.fit(n=self.n, callbacks=[tracker])

        logging.info('Done!')
        logging.info('\n====================')

        return sample, tracker

    def run_nfvi(self, setting=None):
        logging.info('\n====================')
        logging.info('running Norm Flow VI..')

        if setting is None:
            setting = self.setting['NFVI']

        with self.model:
            vi = pm.NFVI(start=self.start, **setting)
            sample = vi.fit(n=self.n)

        logging.info('Done!')
        logging.info('\n====================')

        return sample

    def run_opvi(self, setting=None):
        raise NotImplementedError

    def run_svgd(self, setting=None):
        raise NotImplementedError



