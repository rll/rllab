


import numpy as np
from rllab.core.serializable import Serializable


class ProductRegressor(Serializable):
    """
    A class for performing MLE regression by fitting a product distribution to the outputs. A separate regressor will
    be trained for each individual input distribution.
    """

    def __init__(self, regressors):
        """
        :param regressors: List of individual regressors
        """
        Serializable.quick_init(self, locals())
        self.regressors = regressors
        self.output_dims = [x.output_dim for x in regressors]

    def _split_ys(self, ys):
        ys = np.asarray(ys)
        split_ids = np.cumsum(self.output_dims)[:-1]
        return np.split(ys, split_ids, axis=1)

    def fit(self, xs, ys):
        for regressor, split_ys in zip(self.regressors, self._split_ys(ys)):
            regressor.fit(xs, split_ys)

    def predict(self, xs):
        return np.concatenate([
            regressor.predict(xs) for regressor in self.regressors
        ], axis=1)

    def sample_predict(self, xs):
        return np.concatenate([
            regressor.sample_predict(xs) for regressor in self.regressors
        ], axis=1)

    def predict_log_likelihood(self, xs, ys):
        return np.sum([
                          regressor.predict_log_likelihood(xs, split_ys)
                          for regressor, split_ys in zip(self.regressors, self._split_ys(ys))
                          ], axis=0)

    def get_param_values(self, **tags):
        return np.concatenate(
            [regressor.get_param_values(**tags) for regressor in self.regressors]
        )

    def set_param_values(self, flattened_params, **tags):
        param_dims = [
            np.prod(regressor.get_param_shapes(**tags))
            for regressor in self.regressors
            ]
        split_ids = np.cumsum(param_dims)[:-1]
        for regressor, split_param_values in zip(self.regressors, np.split(flattened_params, split_ids)):
            regressor.set_param_values(split_param_values)
