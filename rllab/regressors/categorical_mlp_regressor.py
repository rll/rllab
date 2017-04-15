import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import theano
import theano.tensor as TT

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.core.serializable import Serializable
from rllab.distributions.categorical import Categorical
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc import special
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer

NONE = list()


class CategoricalMLPRegressor(LasagnePowered):
    """
    A class for performing regression (or classification, really) by fitting a categorical distribution to the outputs.
    Assumes that the outputs will be always a one hot vector.
    """

    def __init__(
            self,
            input_shape,
            output_dim,
            prob_network=None,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.rectify,
            optimizer=None,
            use_trust_region=True,
            step_size=0.01,
            normalize_inputs=True,
            name=None,
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

        if optimizer is None:
            if use_trust_region:
                optimizer = PenaltyLbfgsOptimizer()
            else:
                optimizer = LbfgsOptimizer()

        self.output_dim = output_dim
        self._optimizer = optimizer

        if prob_network is None:
            prob_network = MLP(
                input_shape=input_shape,
                output_dim=output_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=NL.softmax,
            )

        l_prob = prob_network.output_layer

        LasagnePowered.__init__(self, [l_prob])

        xs_var = prob_network.input_layer.input_var
        ys_var = TT.imatrix("ys")
        old_prob_var = TT.matrix("old_prob")

        x_mean_var = theano.shared(
            np.zeros((1,) + input_shape),
            name="x_mean",
            broadcastable=(True,) + (False,) * len(input_shape)
        )
        x_std_var = theano.shared(
            np.ones((1,) + input_shape),
            name="x_std",
            broadcastable=(True,) + (False,) * len(input_shape)
        )

        normalized_xs_var = (xs_var - x_mean_var) / x_std_var

        prob_var = L.get_output(l_prob, {prob_network.input_layer: normalized_xs_var})

        old_info_vars = dict(prob=old_prob_var)
        info_vars = dict(prob=prob_var)

        dist = self._dist = Categorical(output_dim)

        mean_kl = TT.mean(dist.kl_sym(old_info_vars, info_vars))

        loss = - TT.mean(dist.log_likelihood_sym(ys_var, info_vars))

        predicted = special.to_onehot_sym(TT.argmax(prob_var, axis=1), output_dim)

        self._f_predict = ext.compile_function([xs_var], predicted)
        self._f_prob = ext.compile_function([xs_var], prob_var)
        self._prob_network = prob_network
        self._l_prob = l_prob

        optimizer_args = dict(
            loss=loss,
            target=self,
            network_outputs=[prob_var],
        )

        if use_trust_region:
            optimizer_args["leq_constraint"] = (mean_kl, step_size)
            optimizer_args["inputs"] = [xs_var, ys_var, old_prob_var]
        else:
            optimizer_args["inputs"] = [xs_var, ys_var]

        self._optimizer.update_opt(**optimizer_args)

        self._use_trust_region = use_trust_region
        self._name = name

        self._normalize_inputs = normalize_inputs
        self._x_mean_var = x_mean_var
        self._x_std_var = x_std_var

    def fit(self, xs, ys):
        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self._x_mean_var.set_value(np.mean(xs, axis=0, keepdims=True))
            self._x_std_var.set_value(np.std(xs, axis=0, keepdims=True) + 1e-8)
        if self._use_trust_region:
            old_prob = self._f_prob(xs)
            inputs = [xs, ys, old_prob]
        else:
            inputs = [xs, ys]
        loss_before = self._optimizer.loss(inputs)
        if self._name:
            prefix = self._name + "_"
        else:
            prefix = ""
        logger.record_tabular(prefix + 'LossBefore', loss_before)
        self._optimizer.optimize(inputs)
        loss_after = self._optimizer.loss(inputs)
        logger.record_tabular(prefix + 'LossAfter', loss_after)
        logger.record_tabular(prefix + 'dLoss', loss_before - loss_after)

    def predict(self, xs):
        return self._f_predict(np.asarray(xs))

    def predict_log_likelihood(self, xs, ys):
        prob = self._f_prob(np.asarray(xs))
        return self._dist.log_likelihood(np.asarray(ys), dict(prob=prob))

    def log_likelihood_sym(self, x_var, y_var):
        normalized_xs_var = (x_var - self._x_mean_var) / self._x_std_var
        prob = L.get_output(self._l_prob, {self._prob_network.input_layer: normalized_xs_var})
        return self._dist.log_likelihood_sym(TT.cast(y_var, 'int32'), dict(prob=prob))

    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)
