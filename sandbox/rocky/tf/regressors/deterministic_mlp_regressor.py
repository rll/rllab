





import numpy as np

import tensorflow as tf
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.optimizers.lbfgs_optimizer import LbfgsOptimizer
import sandbox.rocky.tf.core.layers as L
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger

NONE = list()


class DeterministicMLPRegressor(LayersPowered, Serializable):
    """
    A class for performing nonlinear regression.
    """

    def __init__(
            self,
            name,
            input_shape,
            output_dim,
            network=None,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            optimizer=None,
            normalize_inputs=True,
    ):
        """
        :param input_shape: Shape of the input data.
        :param output_dim: Dimension of output.
        :param hidden_sizes: Number of hidden units of each layer of the mean network.
        :param hidden_nonlinearity: Non-linearity used for each layer of the mean network.
        :param optimizer: Optimizer for minimizing the negative log-likelihood.
        """
        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):

            if optimizer is None:
                optimizer = LbfgsOptimizer(name="optimizer")

            self.output_dim = output_dim
            self.optimizer = optimizer

            if network is None:
                network = MLP(
                    input_shape=input_shape,
                    output_dim=output_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    name="network"
                )

            l_out = network.output_layer

            LayersPowered.__init__(self, [l_out])

            xs_var = network.input_layer.input_var
            ys_var = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name="ys")

            x_mean_var = tf.get_variable(
                name="x_mean",
                shape=(1,) + input_shape,
                initializer=tf.constant_initializer(0., dtype=tf.float32)
            )
            x_std_var = tf.get_variable(
                name="x_std",
                shape=(1,) + input_shape,
                initializer=tf.constant_initializer(1., dtype=tf.float32)
            )

            normalized_xs_var = (xs_var - x_mean_var) / x_std_var

            fit_ys_var = L.get_output(l_out, {network.input_layer: normalized_xs_var})

            loss = - tf.reduce_mean(tf.square(fit_ys_var - ys_var))

            self.f_predict = tensor_utils.compile_function([xs_var], fit_ys_var)

            optimizer_args = dict(
                loss=loss,
                target=self,
                network_outputs=[fit_ys_var],
            )

            optimizer_args["inputs"] = [xs_var, ys_var]

            self.optimizer.update_opt(**optimizer_args)

            self.name = name
            self.l_out = l_out

            self.normalize_inputs = normalize_inputs
            self.x_mean_var = x_mean_var
            self.x_std_var = x_std_var

    def predict_sym(self, xs):
        return L.get_output(self.l_out, xs)

    # def fit(self, xs, ys):
    #     if self._normalize_inputs:
    #         # recompute normalizing constants for inputs
    #         new_mean = np.mean(xs, axis=0, keepdims=True)
    #         new_std = np.std(xs, axis=0, keepdims=True) + 1e-8
    #         tf.get_default_session().run(tf.group(
    #             tf.assign(self._x_mean_var, new_mean),
    #             tf.assign(self._x_std_var, new_std),
    #         ))
    #         inputs = [xs, ys]
    #     loss_before = self._optimizer.loss(inputs)
    #     if self._name:
    #         prefix = self._name + "_"
    #     else:
    #         prefix = ""
    #     logger.record_tabular(prefix + 'LossBefore', loss_before)
    #     self._optimizer.optimize(inputs)
    #     loss_after = self._optimizer.loss(inputs)
    #     logger.record_tabular(prefix + 'LossAfter', loss_after)
    #     logger.record_tabular(prefix + 'dLoss', loss_before - loss_after)

    def predict(self, xs):
        return self.f_predict(np.asarray(xs))

    def get_param_values(self, **tags):
        return LayersPowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        return LayersPowered.set_param_values(self, flattened_params, **tags)
