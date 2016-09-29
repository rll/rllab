# encoding: utf-8

import lasagne.layers as L
import lasagne
import theano
import theano.tensor as TT


class ParamLayer(L.Layer):

    def __init__(self, incoming, num_units, param=lasagne.init.Constant(0.),
                 trainable=True, **kwargs):
        super(ParamLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.param = self.add_param(
            param,
            (num_units,),
            name="param",
            trainable=trainable
        )

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        ndim = input.ndim
        reshaped_param = TT.reshape(self.param, (1,) * (ndim - 1) + (self.num_units,))
        tile_arg = TT.concatenate([input.shape[:-1], [1]])
        tiled = TT.tile(reshaped_param, tile_arg, ndim=ndim)
        return tiled


class OpLayer(L.MergeLayer):
    def __init__(self, incoming, op,
                 shape_op=lambda x: x, extras=None, **kwargs):
        if extras is None:
            extras = []
        incomings = [incoming] + extras
        super(OpLayer, self).__init__(incomings, **kwargs)
        self.op = op
        self.shape_op = shape_op
        self.incomings = incomings

    def get_output_shape_for(self, input_shapes):
        return self.shape_op(*input_shapes)

    def get_output_for(self, inputs, **kwargs):
        return self.op(*inputs)


class BatchNormLayer(L.Layer):
    """
    lasagne.layers.BatchNormLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.1, mode='low_mem',
    beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), std=lasagne.init.Constant(1), **kwargs)

    Batch Normalization

    This layer implements batch normalization of its inputs, following [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed. The crucial part is that the mean and variance are
    computed across the batch dimension, i.e., over examples, not per example.

    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`
    (nota bene: instead of :math:`\\sigma^2`, the layer actually stores
    :math:`1 / \\sqrt{\\sigma^2 + \\epsilon}`, for compatibility to cuDNN).
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.

    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.

    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.

    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.

    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.

    In case you set `axes` to not include the batch dimension (the first axis,
    usually), normalization is done per example, not across examples. This does
    not require any averages, so you can pass ``batch_norm_update_averages``
    and ``batch_norm_use_averages`` as ``False`` in this case.

    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 mode='low_mem', beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
                 mean=lasagne.init.Constant(0), std=lasagne.init.Constant(1), **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha
        self.mode = mode

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=False)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.std = self.add_param(std, shape, 'std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        input_mean = input.mean(self.axes)
        input_std = TT.sqrt(input.var(self.axes) + self.epsilon)

        # Decide whether to use the stored averages or mini-batch statistics
        use_averages = kwargs.get('batch_norm_use_averages',
                                  deterministic)
        if use_averages:
            mean = self.mean
            std = self.std
        else:
            mean = input_mean
            std = input_std

        # Decide whether to update the stored averages
        update_averages = kwargs.get('batch_norm_update_averages',
                                     not deterministic)
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_std.default_update = ((1 - self.alpha) *
                                              running_std +
                                              self.alpha * input_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            std += 0 * running_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(list(range(input.ndim - len(self.axes))))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        std = std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * TT.inv(std)) + beta
        return normalized


def batch_norm(layer, **kwargs):
    """
    Apply batch normalization to an existing layer. This is a convenience
    function modifying an existing layer to include batch normalization: It
    will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), remove
    the layer's bias if there is one (because it would be redundant), and add
    a :class:`BatchNormLayer` and :class:`NonlinearityLayer` on top.

    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        irreversibly modified as specified above
    **kwargs
        Any additional keyword arguments are passed on to the
        :class:`BatchNormLayer` constructor.

    Returns
    -------
    BatchNormLayer or NonlinearityLayer instance
        A batch normalization layer stacked on the given modified `layer`, or
        a nonlinearity layer stacked on top of both if `layer` was nonlinear.

    Examples
    --------
    Just wrap any layer into a :func:`batch_norm` call on creating it:

    >>> from lasagne.layers import InputLayer, DenseLayer, batch_norm
    >>> from lasagne.nonlinearities import tanh
    >>> l1 = InputLayer((64, 768))
    >>> l2 = batch_norm(DenseLayer(l1, num_units=500, nonlinearity=tanh))

    This introduces batch normalization right before its nonlinearity:

    >>> from lasagne.layers import get_all_layers
    >>> [l.__class__.__name__ for l in get_all_layers(l2)]
    ['InputLayer', 'DenseLayer', 'BatchNormLayer', 'NonlinearityLayer']
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    layer = BatchNormLayer(layer, **kwargs)
    if nonlinearity is not None:
        layer = L.NonlinearityLayer(layer, nonlinearity)
    return layer
