import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.regressors.gaussian_conv_regressor import GaussianConvRegressor


class GaussianConvBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianConvBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianConvRegressor(
            input_shape=env_spec.observation_space.shape,
            output_dim=1,
            name="vf",
            **regressor_args
        )

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
