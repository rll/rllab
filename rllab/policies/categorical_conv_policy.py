from rllab.core.lasagne_powered import LasagnePowered
import lasagne.layers as L
from rllab.core.network import ConvNetwork
from rllab.distributions.categorical import Categorical
from rllab.policies.base import StochasticPolicy
from rllab.misc import tensor_utils
from rllab.spaces.discrete import Discrete

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.overrides import overrides
import numpy as np
import lasagne.nonlinearities as NL


class CategoricalConvPolicy(StochasticPolicy, LasagnePowered):
    def __init__(
            self,
            name,
            env_spec,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_sizes=[],
            hidden_nonlinearity=NL.rectify,
            output_nonlinearity=NL.softmax,
            prob_network=None,
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        self._env_spec = env_spec

        if prob_network is None:
            prob_network = ConvNetwork(
                input_shape=env_spec.observation_space.shape,
                output_dim=env_spec.action_space.n,
                conv_filters=conv_filters,
                conv_filter_sizes=conv_filter_sizes,
                conv_strides=conv_strides,
                conv_pads=conv_pads,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=NL.softmax,
                name="prob_network",
            )

        self._l_prob = prob_network.output_layer
        self._l_obs = prob_network.input_layer
        self._f_prob = ext.compile_function(
            [prob_network.input_layer.input_var],
            L.get_output(prob_network.output_layer)
        )

        self._dist = Categorical(env_spec.action_space.n)

        super(CategoricalConvPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [prob_network.output_layer])

    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        return dict(
            prob=L.get_output(
                self._l_prob,
                {self._l_obs: obs_var}
            )
        )

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self._f_prob(obs))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist
