import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init
import numpy as np
import theano.tensor as TT

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import GRUNetwork
from rllab.core.serializable import Serializable
from rllab.distributions.recurrent_diagonal_gaussian import RecurrentDiagonalGaussian
from rllab.misc import ext
from rllab.spaces import Discrete
from rllab.misc.overrides import overrides
from rllab.policies.base import StochasticPolicy


class GaussianGRUPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32,),
            state_include_action=True,
            hidden_nonlinearity=NL.tanh,
            learn_std=True,
            init_std=1.0,
            output_nonlinearity=None,
    ):
        """
        :param env_spec: A spec for the env.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        Serializable.quick_init(self, locals())
        super(GaussianGRUPolicy, self).__init__(env_spec)

        assert len(hidden_sizes) == 1

        if state_include_action:
            obs_dim = env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim
        else:
            obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        mean_network = GRUNetwork(
            input_shape=(obs_dim,),
            output_dim=action_dim,
            hidden_dim=hidden_sizes[0],
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=NL.softmax,
        )

        l_mean = mean_network.output_layer
        obs_var = mean_network.input_var

        l_log_std = ParamLayer(
            mean_network.input_layer,
            num_units=action_dim,
            param=lasagne.init.Constant(np.log(init_std)),
            name="output_log_std",
            trainable=learn_std,
        )

        l_step_log_std = ParamLayer(
            mean_network.step_input_layer,
            num_units=action_dim,
            param=l_log_std.param,
            name="step_output_log_std",
            trainable=learn_std,
        )

        self._mean_network = mean_network
        self._l_log_std = l_log_std
        self._state_include_action = state_include_action

        self._f_step_mean_std = ext.compile_function(
            [
                mean_network.step_input_layer.input_var,
                mean_network.step_prev_hidden_layer.input_var
            ],
            L.get_output([
                mean_network.step_output_layer,
                l_step_log_std,
                mean_network.step_hidden_layer
            ])
        )

        self._prev_action = None
        self._prev_hidden = None
        self._hidden_sizes = hidden_sizes
        self._dist = RecurrentDiagonalGaussian(action_dim)

        self.reset()

        LasagnePowered.__init__(self, [mean_network.output_layer, l_log_std])

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        n_batches, n_steps = obs_var.shape[:2]
        obs_var = obs_var.reshape((n_batches, n_steps, -1))
        if self._state_include_action:
            prev_action_var = state_info_vars["prev_action"]
            all_input_var = TT.concatenate(
                [obs_var, prev_action_var],
                axis=2
            )
        else:
            all_input_var = obs_var
        means, log_stds = L.get_output([self._mean_network.output_layer, self._l_log_std], all_input_var)
        return dict(mean=means, log_std=log_stds)

    def reset(self):
        self._prev_action = None
        self._prev_hidden = self._mean_network.hid_init_param.get_value()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        if self._state_include_action:
            if self._prev_action is None:
                prev_action = np.zeros((self.action_space.flat_dim,))
            else:
                prev_action = self.action_space.flatten(self._prev_action)
            all_input = np.concatenate([
                self.observation_space.flatten(observation),
                prev_action
            ])
        else:
            all_input = self.observation_space.flatten(observation)
            # should not be used
            prev_action = np.nan
        mean, log_std, hidden_vec = [x[0] for x in self._f_step_mean_std([all_input], [self._prev_hidden])]
        rnd = np.random.randn(len(mean))
        action = rnd * np.exp(log_std) + mean
        self._prev_action = action
        self._prev_hidden = hidden_vec
        agent_info = dict(mean=mean, log_std=log_std)
        if self._state_include_action:
            agent_info["prev_action"] = prev_action
        return action, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self._dist

    @property
    def state_info_keys(self):
        if self._state_include_action:
            return ["prev_action"]
        else:
            return []
