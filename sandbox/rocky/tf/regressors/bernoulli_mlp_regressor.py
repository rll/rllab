


import sandbox.rocky.tf.core.layers as L
import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import MLP
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.distributions.bernoulli import Bernoulli
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import logger
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.lbfgs_optimizer import LbfgsOptimizer


class BernoulliMLPRegressor(LayersPowered, Serializable):
    """
    A class for performing regression (or classification, really) by fitting a bernoulli distribution to each of the
    output units.
    """

    def __init__(
            self,
            input_shape,
            output_dim,
            name,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            optimizer=None,
            tr_optimizer=None,
            use_trust_region=True,
            step_size=0.01,
            normalize_inputs=True,
            no_initial_trust_region=True,
    ):
        """
        :param input_shape: Shape of the input data.
        :param output_dim: Dimension of output.
        :param hidden_sizes: Number of hidden units of each layer of the mean network.
        :param hidden_nonlinearity: Non-linearity used for each layer of the mean network.
        :param optimizer: Optimizer for minimizing the negative log-likelihood.
        :param use_trust_region: Whether to use trust region constraint.
        :param step_size: KL divergence constraint for each iteration
        """
        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):

            if optimizer is None:
                optimizer = LbfgsOptimizer(name="optimizer")
            if tr_optimizer is None:
                tr_optimizer = ConjugateGradientOptimizer()

            self.output_dim = output_dim
            self.optimizer = optimizer
            self.tr_optimizer = tr_optimizer

            p_network = MLP(
                input_shape=input_shape,
                output_dim=output_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=tf.nn.sigmoid,
                name="p_network"
            )

            l_p = p_network.output_layer

            LayersPowered.__init__(self, [l_p])

            xs_var = p_network.input_layer.input_var
            ys_var = tf.placeholder(dtype=tf.float32, shape=(None, output_dim), name="ys")
            old_p_var = tf.placeholder(dtype=tf.float32, shape=(None, output_dim), name="old_p")

            x_mean_var = tf.get_variable(name="x_mean", initializer=tf.zeros_initializer(), shape=(1,) + input_shape)
            x_std_var = tf.get_variable(name="x_std", initializer=tf.ones_initializer(), shape=(1,) + input_shape)

            normalized_xs_var = (xs_var - x_mean_var) / x_std_var

            p_var = L.get_output(l_p, {p_network.input_layer: normalized_xs_var})

            old_info_vars = dict(p=old_p_var)
            info_vars = dict(p=p_var)

            dist = self._dist = Bernoulli(output_dim)

            mean_kl = tf.reduce_mean(dist.kl_sym(old_info_vars, info_vars))

            loss = - tf.reduce_mean(dist.log_likelihood_sym(ys_var, info_vars))

            predicted = p_var >= 0.5

            self.f_predict = tensor_utils.compile_function([xs_var], predicted)
            self.f_p = tensor_utils.compile_function([xs_var], p_var)
            self.l_p = l_p

            self.optimizer.update_opt(loss=loss, target=self, network_outputs=[p_var], inputs=[xs_var, ys_var])
            self.tr_optimizer.update_opt(loss=loss, target=self, network_outputs=[p_var],
                                         inputs=[xs_var, ys_var, old_p_var],
                                         leq_constraint=(mean_kl, step_size)
                                         )

            self.use_trust_region = use_trust_region
            self.name = name

            self.normalize_inputs = normalize_inputs
            self.x_mean_var = x_mean_var
            self.x_std_var = x_std_var
            self.first_optimized = not no_initial_trust_region

    def fit(self, xs, ys):
        if self.normalize_inputs:
            # recompute normalizing constants for inputs
            new_mean = np.mean(xs, axis=0, keepdims=True)
            new_std = np.std(xs, axis=0, keepdims=True) + 1e-8
            tf.get_default_session().run(tf.group(
                tf.assign(self.x_mean_var, new_mean),
                tf.assign(self.x_std_var, new_std),
            ))
            # self._x_mean_var.set_value(np.mean(xs, axis=0, keepdims=True))
            # self._x_std_var.set_value(np.std(xs, axis=0, keepdims=True) + 1e-8)
        if self.use_trust_region and self.first_optimized:
            old_p = self.f_p(xs)
            inputs = [xs, ys, old_p]
            optimizer = self.tr_optimizer
        else:
            inputs = [xs, ys]
            optimizer = self.optimizer
        loss_before = optimizer.loss(inputs)
        if self.name:
            prefix = self.name + "_"
        else:
            prefix = ""
        logger.record_tabular(prefix + 'LossBefore', loss_before)
        optimizer.optimize(inputs)
        loss_after = optimizer.loss(inputs)
        logger.record_tabular(prefix + 'LossAfter', loss_after)
        logger.record_tabular(prefix + 'dLoss', loss_before - loss_after)
        self.first_optimized = True

    def predict(self, xs):
        return self.f_predict(np.asarray(xs))

    def sample_predict(self, xs):
        p = self.f_p(xs)
        return self._dist.sample(dict(p=p))

    def predict_log_likelihood(self, xs, ys):
        p = self.f_p(np.asarray(xs))
        return self._dist.log_likelihood(np.asarray(ys), dict(p=p))

    def get_param_values(self, **tags):
        return LayersPowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        return LayersPowered.set_param_values(self, flattened_params, **tags)
