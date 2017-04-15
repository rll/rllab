import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import theano
import theano.tensor as TT

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.ext import compile_function
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.misc.ext import iterate_minibatches_generic


class GaussianMLPRegressor(LasagnePowered):
    """
    A class for performing regression by fitting a Gaussian distribution to the outputs.
    """

    def __init__(
            self,
            input_shape,
            output_dim,
            mean_network=None,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.rectify,
            optimizer=None,
            use_trust_region=True,
            step_size=0.01,
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            std_nonlinearity=None,
            normalize_inputs=True,
            normalize_outputs=True,
            name=None,
            batchsize=None,
            subsample_factor=1.,
    ):
        """
        :param input_shape: Shape of the input data.
        :param output_dim: Dimension of output.
        :param hidden_sizes: Number of hidden units of each layer of the mean network.
        :param hidden_nonlinearity: Non-linearity used for each layer of the mean network.
        :param optimizer: Optimizer for minimizing the negative log-likelihood.
        :param use_trust_region: Whether to use trust region constraint.
        :param step_size: KL divergence constraint for each iteration
        :param learn_std: Whether to learn the standard deviations. Only effective if adaptive_std is False. If
        adaptive_std is True, this parameter is ignored, and the weights for the std network are always learned.
        :param adaptive_std: Whether to make the std a function of the states.
        :param std_share_network: Whether to use the same network as the mean.
        :param std_hidden_sizes: Number of hidden units of each layer of the std network. Only used if
        `std_share_network` is False. It defaults to the same architecture as the mean.
        :param std_nonlinearity: Non-linearity used for each layer of the std network. Only used if `std_share_network`
        is False. It defaults to the same non-linearity as the mean.
        """
        Serializable.quick_init(self, locals())

        self._batchsize = batchsize
        self._subsample_factor = subsample_factor

        if optimizer is None:
            if use_trust_region:
                optimizer = PenaltyLbfgsOptimizer()
            else:
                optimizer = LbfgsOptimizer()

        self._optimizer = optimizer

        if mean_network is None:
            mean_network = MLP(
                input_shape=input_shape,
                output_dim=output_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None,
            )

        l_mean = mean_network.output_layer

        if adaptive_std:
            l_log_std = MLP(
                input_shape=input_shape,
                input_var=mean_network.input_layer.input_var,
                output_dim=output_dim,
                hidden_sizes=std_hidden_sizes,
                hidden_nonlinearity=std_nonlinearity,
                output_nonlinearity=None,
            ).output_layer
        else:
            l_log_std = ParamLayer(
                mean_network.input_layer,
                num_units=output_dim,
                param=lasagne.init.Constant(np.log(init_std)),
                name="output_log_std",
                trainable=learn_std,
            )

        LasagnePowered.__init__(self, [l_mean, l_log_std])

        xs_var = mean_network.input_layer.input_var
        ys_var = TT.matrix("ys")
        old_means_var = TT.matrix("old_means")
        old_log_stds_var = TT.matrix("old_log_stds")

        x_mean_var = theano.shared(
            np.zeros((1,) + input_shape, dtype=theano.config.floatX),
            name="x_mean",
            broadcastable=(True,) + (False,) * len(input_shape)
        )
        x_std_var = theano.shared(
            np.ones((1,) + input_shape, dtype=theano.config.floatX),
            name="x_std",
            broadcastable=(True,) + (False,) * len(input_shape)
        )
        y_mean_var = theano.shared(
            np.zeros((1, output_dim), dtype=theano.config.floatX),
            name="y_mean",
            broadcastable=(True, False)
        )
        y_std_var = theano.shared(
            np.ones((1, output_dim), dtype=theano.config.floatX),
            name="y_std",
            broadcastable=(True, False)
        )

        normalized_xs_var = (xs_var - x_mean_var) / x_std_var
        normalized_ys_var = (ys_var - y_mean_var) / y_std_var

        normalized_means_var = L.get_output(
            l_mean, {mean_network.input_layer: normalized_xs_var})
        normalized_log_stds_var = L.get_output(
            l_log_std, {mean_network.input_layer: normalized_xs_var})

        means_var = normalized_means_var * y_std_var + y_mean_var
        log_stds_var = normalized_log_stds_var + TT.log(y_std_var)

        normalized_old_means_var = (old_means_var - y_mean_var) / y_std_var
        normalized_old_log_stds_var = old_log_stds_var - TT.log(y_std_var)

        dist = self._dist = DiagonalGaussian(output_dim)

        normalized_dist_info_vars = dict(
            mean=normalized_means_var, log_std=normalized_log_stds_var)

        mean_kl = TT.mean(dist.kl_sym(
            dict(mean=normalized_old_means_var,
                 log_std=normalized_old_log_stds_var),
            normalized_dist_info_vars,
        ))

        loss = - \
            TT.mean(dist.log_likelihood_sym(
                normalized_ys_var, normalized_dist_info_vars))

        self._f_predict = compile_function([xs_var], means_var)
        self._f_pdists = compile_function([xs_var], [means_var, log_stds_var])
        self._l_mean = l_mean
        self._l_log_std = l_log_std

        optimizer_args = dict(
            loss=loss,
            target=self,
            network_outputs=[normalized_means_var, normalized_log_stds_var],
        )

        if use_trust_region:
            optimizer_args["leq_constraint"] = (mean_kl, step_size)
            optimizer_args["inputs"] = [
                xs_var, ys_var, old_means_var, old_log_stds_var]
        else:
            optimizer_args["inputs"] = [xs_var, ys_var]

        self._optimizer.update_opt(**optimizer_args)

        self._use_trust_region = use_trust_region
        self._name = name

        self._normalize_inputs = normalize_inputs
        self._normalize_outputs = normalize_outputs
        self._mean_network = mean_network
        self._x_mean_var = x_mean_var
        self._x_std_var = x_std_var
        self._y_mean_var = y_mean_var
        self._y_std_var = y_std_var

    def fit(self, xs, ys):

        if self._subsample_factor < 1:
            num_samples_tot = xs.shape[0]
            idx = np.random.randint(0, num_samples_tot, int(num_samples_tot * self._subsample_factor))
            xs, ys = xs[idx], ys[idx]

        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self._x_mean_var.set_value(
                np.mean(xs, axis=0, keepdims=True).astype(theano.config.floatX))
            self._x_std_var.set_value(
                (np.std(xs, axis=0, keepdims=True) + 1e-8).astype(theano.config.floatX))
        if self._normalize_outputs:
            # recompute normalizing constants for outputs
            self._y_mean_var.set_value(
                np.mean(ys, axis=0, keepdims=True).astype(theano.config.floatX))
            self._y_std_var.set_value(
                (np.std(ys, axis=0, keepdims=True) + 1e-8).astype(theano.config.floatX))
        if self._name:
            prefix = self._name + "_"
        else:
            prefix = ""
        # FIXME: needs batch computation to avoid OOM.
        loss_before, loss_after, mean_kl, batch_count = 0., 0., 0., 0
        for batch in iterate_minibatches_generic(input_lst=[xs, ys], batchsize=self._batchsize, shuffle=True):
            batch_count += 1
            xs, ys = batch
            if self._use_trust_region:
                old_means, old_log_stds = self._f_pdists(xs)
                inputs = [xs, ys, old_means, old_log_stds]
            else:
                inputs = [xs, ys]
            loss_before += self._optimizer.loss(inputs)

            self._optimizer.optimize(inputs)
            loss_after += self._optimizer.loss(inputs)
            if self._use_trust_region:
                mean_kl += self._optimizer.constraint_val(inputs)

        logger.record_tabular(prefix + 'LossBefore', loss_before / batch_count)
        logger.record_tabular(prefix + 'LossAfter', loss_after / batch_count)
        logger.record_tabular(prefix + 'dLoss', loss_before - loss_after / batch_count)
        if self._use_trust_region:
            logger.record_tabular(prefix + 'MeanKL', mean_kl / batch_count)

    def predict(self, xs):
        """
        Return the maximum likelihood estimate of the predicted y.
        :param xs:
        :return:
        """
        return self._f_predict(xs)

    def sample_predict(self, xs):
        """
        Sample one possible output from the prediction distribution.
        :param xs:
        :return:
        """
        means, log_stds = self._f_pdists(xs)
        return self._dist.sample(dict(mean=means, log_std=log_stds))

    def predict_log_likelihood(self, xs, ys):
        means, log_stds = self._f_pdists(xs)
        return self._dist.log_likelihood(ys, dict(mean=means, log_std=log_stds))

    def log_likelihood_sym(self, x_var, y_var):
        normalized_xs_var = (x_var - self._x_mean_var) / self._x_std_var

        normalized_means_var, normalized_log_stds_var = \
            L.get_output([self._l_mean, self._l_log_std], {
                self._mean_network.input_layer: normalized_xs_var})

        means_var = normalized_means_var * self._y_std_var + self._y_mean_var
        log_stds_var = normalized_log_stds_var + TT.log(self._y_std_var)

        return self._dist.log_likelihood_sym(y_var, dict(mean=means_var, log_std=log_stds_var))

    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)
