import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import theano.tensor as TT

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import GRUNetwork
from rllab.core.serializable import Serializable
from rllab.distributions.recurrent_categorical import RecurrentCategorical
from rllab.misc import ext
from rllab.spaces import Discrete
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.policies.base import StochasticPolicy


class CategoricalGRUPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32,),
            state_include_action=True,
            hidden_nonlinearity=NL.tanh):
        """
        :param env_spec: A spec for the env.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        super(CategoricalGRUPolicy, self).__init__(env_spec)

        assert len(hidden_sizes) == 1

        if state_include_action:
            input_shape = (env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim,)
        else:
            input_shape = (env_spec.observation_space.flat_dim,)

        prob_network = GRUNetwork(
            input_shape=input_shape,
            output_dim=env_spec.action_space.n,
            hidden_dim=hidden_sizes[0],
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=NL.softmax,
        )

        self._prob_network = prob_network
        self._state_include_action = state_include_action

        self._f_step_prob = ext.compile_function(
            [
                prob_network.step_input_layer.input_var,
                prob_network.step_prev_hidden_layer.input_var
            ],
            L.get_output([
                prob_network.step_output_layer,
                prob_network.step_hidden_layer
            ])
        )

        self._prev_action = None
        self._prev_hidden = None
        self._hidden_sizes = hidden_sizes
        self._dist = RecurrentCategorical()

        self.reset()

        LasagnePowered.__init__(self, [prob_network.output_layer])

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
        return dict(
            prob=L.get_output(
                self._prob_network.output_layer,
                {self._prob_network.input_layer: all_input_var}
            )
        )

    def reset(self):
        self._prev_action = None
        self._prev_hidden = self._prob_network.hid_init_param.get_value()

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
        probs, hidden_vec = [x[0] for x in self._f_step_prob([all_input], [self._prev_hidden])]
        action = special.weighted_sample(probs, xrange(self.action_space.n))
        self._prev_action = action
        self._prev_hidden = hidden_vec
        agent_info = dict(prob=probs)
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
