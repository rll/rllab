import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import theano.tensor as TT

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import GRUNetwork
from rllab.core.lasagne_layers import OpLayer
from rllab.core.serializable import Serializable
from rllab.distributions.recurrent_categorical import RecurrentCategorical
from rllab.misc import ext
from rllab.spaces import Discrete
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.policies.base import StochasticPolicy


class CategoricalGRUPolicy(StochasticPolicy, LasagnePowered):
    def __init__(
            self,
            env_spec,
            hidden_dim=32,
            feature_network=None,
            state_include_action=True,
            hidden_nonlinearity=NL.tanh):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        super(CategoricalGRUPolicy, self).__init__(env_spec)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        if state_include_action:
            input_dim = obs_dim + action_dim
        else:
            input_dim = obs_dim

        l_input = L.InputLayer(
            shape=(None, None, input_dim),
            name="input"
        )

        if feature_network is None:
            feature_dim = input_dim
            l_flat_feature = None
            l_feature = l_input
        else:
            feature_dim = feature_network.output_layer.output_shape[-1]
            l_flat_feature = feature_network.output_layer
            l_feature = OpLayer(
                l_flat_feature,
                extras=[l_input],
                name="reshape_feature",
                op=lambda flat_feature, input: TT.reshape(
                    flat_feature,
                    [input.shape[0], input.shape[1], feature_dim]
                ),
                shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], feature_dim)
            )

        prob_network = GRUNetwork(
            input_shape=(feature_dim,),
            input_layer=l_feature,
            output_dim=env_spec.action_space.n,
            hidden_dim=hidden_dim,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=TT.nnet.softmax,
            name="prob_network"
        )

        self.prob_network = prob_network
        self.feature_network = feature_network
        self.l_input = l_input
        self.state_include_action = state_include_action

        flat_input_var = TT.matrix("flat_input")
        if feature_network is None:
            feature_var = flat_input_var
        else:
            feature_var = L.get_output(l_flat_feature, {feature_network.input_layer: flat_input_var})

        self.f_step_prob = ext.compile_function(
            [
                flat_input_var,
                prob_network.step_prev_hidden_layer.input_var
            ],
            L.get_output([
                prob_network.step_output_layer,
                prob_network.step_hidden_layer
            ], {prob_network.step_input_layer: feature_var})
        )

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.prev_action = None
        self.prev_hidden = None
        self.dist = RecurrentCategorical(env_spec.action_space.n)

        out_layers = [prob_network.output_layer]
        if feature_network is not None:
            out_layers.append(feature_network.output_layer)

        LasagnePowered.__init__(self, out_layers)

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        n_batches, n_steps = obs_var.shape[:2]
        obs_var = obs_var.reshape((n_batches, n_steps, -1))
        if self.state_include_action:
            prev_action_var = state_info_vars["prev_action"]
            all_input_var = TT.concatenate(
                [obs_var, prev_action_var],
                axis=2
            )
        else:
            all_input_var = obs_var

        if self.feature_network is None:
            return dict(
                prob=L.get_output(
                    self.prob_network.output_layer,
                    {self.l_input: all_input_var}
                )
            )
        else:
            flat_input_var = TT.reshape(all_input_var, (-1, self.input_dim))
            return dict(
                prob=L.get_output(
                    self.prob_network.output_layer,
                    {self.l_input: all_input_var, self.feature_network.input_layer: flat_input_var}
                )
            )

    def reset(self):
        self.prev_action = None
        self.prev_hidden = self.prob_network.hid_init_param.get_value()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        if self.state_include_action:
            if self.prev_action is None:
                prev_action = np.zeros((self.action_space.flat_dim,))
            else:
                prev_action = self.action_space.flatten(self.prev_action)
            all_input = np.concatenate([
                self.observation_space.flatten(observation),
                prev_action
            ])
        else:
            all_input = self.observation_space.flatten(observation)
            # should not be used
            prev_action = np.nan
        probs, hidden_vec = [x[0] for x in self.f_step_prob([all_input], [self.prev_hidden])]
        action = special.weighted_sample(probs, range(self.action_space.n))
        self.prev_action = action
        self.prev_hidden = hidden_vec
        agent_info = dict(prob=probs)
        if self.state_include_action:
            agent_info["prev_action"] = prev_action
        return action, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_keys(self):
        if self.state_include_action:
            return ["prev_action"]
        else:
            return []
