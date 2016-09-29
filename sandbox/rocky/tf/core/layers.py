# -*- coding: utf-8 -*-
import functools
import numpy as np
import math
import tensorflow as tf
from tensorflow.python.training import moving_averages
from collections import OrderedDict
from collections import deque
from itertools import chain
from inspect import getargspec
from difflib import get_close_matches
from warnings import warn


class G(object):
    pass


G._n_layers = 0


def create_param(spec, shape, name, trainable=True, regularizable=True):
    if not hasattr(spec, '__call__'):
        assert isinstance(spec, (tf.Tensor, tf.Variable))
        return spec
    assert hasattr(spec, '__call__')
    if regularizable:
        # use the default regularizer
        regularizer = None
    else:
        # do not regularize this variable
        regularizer = lambda _: tf.constant(0.)
    return tf.get_variable(
        name=name, shape=shape, initializer=spec, trainable=trainable,
        regularizer=regularizer, dtype=tf.float32
    )


def as_tuple(x, N, t=None):
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X


def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation
    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.
    Parameters
    ----------
    input_length : int or None
        The size of the input.
    filter_size : int
        The size of the filter.
    stride : int
        The stride of the convolution operation.
    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        both borders.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.
    Returns
    -------
    int or None
        The output size corresponding to the given convolution parameters, or
        ``None`` if `input_size` is ``None``.
    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.
    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length


class Layer(object):
    def __init__(self, incoming, name=None, variable_reuse=None, weight_normalization=False, **kwargs):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming
        self.params = OrderedDict()
        self.weight_normalization = weight_normalization

        if name is None:
            name = "%s_%d" % (type(self).__name__, G._n_layers)
            G._n_layers += 1

        self.name = name
        self.variable_reuse = variable_reuse
        self.get_output_kwargs = []

        if any(d is not None and d <= 0 for d in self.input_shape):
            raise ValueError((
                                 "Cannot create Layer with a non-positive input_shape "
                                 "dimension. input_shape=%r, self.name=%r") % (
                                 self.input_shape, self.name))

    @property
    def output_shape(self):
        shape = self.get_output_shape_for(self.input_shape)
        if any(isinstance(s, (tf.Variable, tf.Tensor)) for s in shape):
            raise ValueError("%s returned a symbolic output shape from its "
                             "get_output_shape_for() method: %r. This is not "
                             "allowed; shapes must be tuples of integers for "
                             "fixed-size dimensions and Nones for variable "
                             "dimensions." % (self.__class__.__name__, shape))
        return shape

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError

    def get_output_for(self, input, **kwargs):
        raise NotImplementedError

    def add_param_plain(self, spec, shape, name, **tags):
        with tf.variable_scope(self.name, reuse=self.variable_reuse):
            tags['trainable'] = tags.get('trainable', True)
            tags['regularizable'] = tags.get('regularizable', True)
            param = create_param(spec, shape, name, **tags)
            self.params[param] = set(tag for tag, value in list(tags.items()) if value)
            return param

    def add_param(self, spec, shape, name, **kwargs):
        param = self.add_param_plain(spec, shape, name, **kwargs)
        if name is not None and name.startswith("W") and self.weight_normalization:
            # Hacky: check if the parameter is a weight matrix. If so, apply weight normalization
            if len(param.get_shape()) == 2:
                v = param
                g = self.add_param_plain(tf.ones_initializer, (shape[1],), name=name + "_wn/g")
                param = v * (tf.reshape(g, (1, -1)) / tf.sqrt(tf.reduce_sum(tf.square(v), 0, keep_dims=True)))
            elif len(param.get_shape()) == 4:
                v = param
                g = self.add_param_plain(tf.ones_initializer, (shape[3],), name=name + "_wn/g")
                param = v * (tf.reshape(g, (1, 1, 1, -1)) / tf.sqrt(tf.reduce_sum(tf.square(v), [0, 1, 2],
                                                                                  keep_dims=True)))
            else:
                raise NotImplementedError
        return param

    def get_params(self, **tags):
        result = list(self.params.keys())

        only = set(tag for tag, value in list(tags.items()) if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - self.params[param])]

        exclude = set(tag for tag, value in list(tags.items()) if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (self.params[param] & exclude)]

        return result


class InputLayer(Layer):
    def __init__(self, shape, input_var=None, **kwargs):
        super(InputLayer, self).__init__(shape, **kwargs)
        self.shape = shape
        if input_var is None:
            if self.name is not None:
                with tf.variable_scope(self.name):
                    input_var = tf.placeholder(tf.float32, shape=shape, name="input")
            else:
                input_var = tf.placeholder(tf.float32, shape=shape, name="input")
        self.input_var = input_var

    @Layer.output_shape.getter
    def output_shape(self):
        return self.shape


class MergeLayer(Layer):
    def __init__(self, incomings, name=None, **kwargs):
        self.input_shapes = [incoming if isinstance(incoming, tuple)
                             else incoming.output_shape
                             for incoming in incomings]
        self.input_layers = [None if isinstance(incoming, tuple)
                             else incoming
                             for incoming in incomings]
        self.name = name
        self.params = OrderedDict()
        self.get_output_kwargs = []

    @Layer.output_shape.getter
    def output_shape(self):
        shape = self.get_output_shape_for(self.input_shapes)
        if any(isinstance(s, (tf.Variable, tf.Tensor)) for s in shape):
            raise ValueError("%s returned a symbolic output shape from its "
                             "get_output_shape_for() method: %r. This is not "
                             "allowed; shapes must be tuples of integers for "
                             "fixed-size dimensions and Nones for variable "
                             "dimensions." % (self.__class__.__name__, shape))
        return shape

    def get_output_shape_for(self, input_shapes):
        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        raise NotImplementedError


class ConcatLayer(MergeLayer):
    """
    Concatenates multiple inputs along the specified axis. Inputs should have
    the same shape except for the dimension specified in axis, which can have
    different sizes.
    Parameters
    -----------
    incomings : a list of :class:`Layer` instances or tuples
        The layers feeding into this layer, or expected input shapes
    axis : int
        Axis which inputs are joined over
    """

    def __init__(self, incomings, axis=1, **kwargs):
        super(ConcatLayer, self).__init__(incomings, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = [next((s for s in sizes if s is not None), None)
                        for sizes in zip(*input_shapes)]

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(i == self.axis or s1 is None or s2 is None or s1 == s2
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "in the concatenation axis")
        # Infer output shape on concatenation axis and return
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        concat_size = None if any(s is None for s in sizes) else sum(sizes)
        output_shape[self.axis] = concat_size
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        dtypes = [x.dtype.as_numpy_dtype for x in inputs]
        if len(set(dtypes)) > 1:
            # need to convert to common data type
            common_dtype = np.core.numerictypes.find_common_type([], dtypes)
            inputs = [tf.cast(x, common_dtype) for x in inputs]
        return tf.concat(concat_dim=self.axis, values=inputs)


concat = ConcatLayer  # shortcut


class XavierUniformInitializer(object):
    def __call__(self, shape, dtype=tf.float32, *args, **kwargs):
        if len(shape) == 2:
            n_inputs, n_outputs = shape
        else:
            receptive_field_size = np.prod(shape[:2])
            n_inputs = shape[-2] * receptive_field_size
            n_outputs = shape[-1] * receptive_field_size
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)


class HeUniformInitializer(object):
    def __call__(self, shape, dtype=tf.float32, *args, **kwargs):
        if len(shape) == 2:
            n_inputs, _ = shape
        else:
            receptive_field_size = np.prod(shape[:2])
            n_inputs = shape[-2] * receptive_field_size
        init_range = math.sqrt(1.0 / n_inputs)
        return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)


def py_ortho_init(scale):
    def _init(shape):
        u, s, v = np.linalg.svd(np.random.uniform(size=shape))
        return np.cast['float32'](u * scale)

    return _init


class OrthogonalInitializer(object):
    def __init__(self, scale=1.1):
        self.scale = scale

    def __call__(self, shape, dtype=tf.float32, *args, **kwargs):
        result, = tf.py_func(py_ortho_init(self.scale), [shape], [tf.float32])
        result.set_shape(shape)
        return result


class ParamLayer(Layer):
    def __init__(self, incoming, num_units, param=tf.zeros_initializer,
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
        ndim = input.get_shape().ndims
        reshaped_param = tf.reshape(self.param, (1,) * (ndim - 1) + (self.num_units,))
        tile_arg = tf.concat(0, [tf.shape(input)[:ndim - 1], [1]])
        tiled = tf.tile(reshaped_param, tile_arg)
        return tiled


class OpLayer(MergeLayer):
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


class DenseLayer(Layer):
    def __init__(self, incoming, num_units, nonlinearity=None, W=XavierUniformInitializer(), b=tf.zeros_initializer,
                 **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = tf.identity if nonlinearity is None else nonlinearity

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.get_shape().ndims > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = tf.reshape(input, tf.pack([tf.shape(input)[0], -1]))
        activation = tf.matmul(input, self.W)
        if self.b is not None:
            activation = activation + tf.expand_dims(self.b, 0)
        return self.nonlinearity(activation)


class BaseConvLayer(Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad="VALID",
                 untie_biases=False,
                 W=XavierUniformInitializer(), b=tf.zeros_initializer,
                 nonlinearity=tf.nn.relu, n=None, **kwargs):
        """
        Input is assumed to be of shape batch*height*width*channels
        """
        super(BaseConvLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = tf.identity
        else:
            self.nonlinearity = nonlinearity

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n + 2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, n, int)
        self.stride = as_tuple(stride, n, int)
        self.untie_biases = untie_biases

        self.pad = pad

        if pad == 'SAME':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = self.output_shape[1:3] + (num_filters,)  # + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.
        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[-1]
        return self.filter_size + (num_input_channels, self.num_filters)

    def get_output_shape_for(self, input_shape):
        if self.pad == 'SAME':
            pad = ('same',) * self.n
        elif self.pad == 'VALID':
            pad = (0,) * self.n
        else:
            import ipdb;
            ipdb.set_trace()
            raise NotImplementedError

        # pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize,) +
                tuple(conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[1:3], self.filter_size,
                             self.stride, pad))) + (self.num_filters,)

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            # raise NotImplementedError
            activation = conved + tf.expand_dims(self.b, 0)
        else:
            activation = conved + tf.reshape(self.b, (1, 1, 1, self.num_filters))

        return self.nonlinearity(activation)

    def convolve(self, input, **kwargs):
        """
        Symbolically convolves `input` with ``self.W``, producing an output of
        shape ``self.output_shape``. To be implemented by subclasses.
        Parameters
        ----------
        input : Theano tensor
            The input minibatch to convolve
        **kwargs
            Any additional keyword arguments from :meth:`get_output_for`
        Returns
        -------
        Theano tensor
            `input` convolved according to the configuration of this layer,
            without any bias or nonlinearity applied.
        """
        raise NotImplementedError("BaseConvLayer does not implement the "
                                  "convolve() method. You will want to "
                                  "use a subclass such as Conv2DLayer.")


class Conv2DLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad="VALID", untie_biases=False,
                 W=XavierUniformInitializer(), b=tf.zeros_initializer,
                 nonlinearity=tf.nn.relu,
                 convolution=tf.nn.conv2d, **kwargs):
        super(Conv2DLayer, self).__init__(incoming=incoming, num_filters=num_filters, filter_size=filter_size,
                                          stride=stride, pad=pad, untie_biases=untie_biases, W=W, b=b,
                                          nonlinearity=nonlinearity, n=2, **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        conved = self.convolution(input, self.W, strides=(1,) + self.stride + (1,), padding=self.pad)
        return conved


def pool_output_length(input_length, pool_size, stride, pad):
    if input_length is None or pool_size is None:
        return None

    if pad == "SAME":
        return int(np.ceil(float(input_length) / float(stride)))

    return int(np.ceil(float(input_length - pool_size + 1) / float(stride)))


class Pool2DLayer(Layer):
    def __init__(self, incoming, pool_size, stride=None, pad="VALID", mode='max', **kwargs):
        super(Pool2DLayer, self).__init__(incoming, **kwargs)

        self.pool_size = as_tuple(pool_size, 2)

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, 2 spatial dimensions, channels)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)

        self.pad = pad

        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[1] = pool_output_length(input_shape[1],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad,
                                             )

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        assert self.mode == "max"
        pooled = tf.nn.max_pool(
            input,
            ksize=(1,) + self.pool_size + (1,),
            strides=(1,) + self.stride + (1,),
            padding=self.pad,
        )
        return pooled


def spatial_expected_softmax(x, temp=1):
    assert len(x.get_shape()) == 4
    vals = []
    for dim in [0, 1]:
        dim_val = x.get_shape()[dim + 1].value
        lin = tf.linspace(-1.0, 1.0, dim_val)
        lin = tf.expand_dims(lin, 1 - dim)
        lin = tf.expand_dims(lin, 0)
        lin = tf.expand_dims(lin, 3)
        m = tf.reduce_max(x, [1, 2], keep_dims=True)
        e = tf.exp((x - m) / temp) + 1e-5
        val = tf.reduce_sum(e * lin, [1, 2]) / (tf.reduce_sum(e, [1, 2]))
        vals.append(tf.expand_dims(val, 2))

    return tf.reshape(tf.concat(2, vals), [-1, x.get_shape()[-1].value * 2])


class SpatialExpectedSoftmaxLayer(Layer):
    """
    Computes the softmax across a spatial region, separately for each channel, followed by an expectation operation.
    """

    def __init__(self, incoming, **kwargs):
        super().__init__(incoming, **kwargs)
        # self.temp = self.add_param(tf.ones_initializer, shape=(), name="temperature")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1] * 2)

    def get_output_for(self, input, **kwargs):
        return spatial_expected_softmax(input)#, self.temp)
        # max_ = tf.reduce_max(input, reduction_indices=[1, 2], keep_dims=True)
        # exp = tf.exp(input - max_) + 1e-5

        # vals = []
        #
        # for dim in [0, 1]:
        #     dim_val = input.get_shape()[dim + 1].value
        #     lin = tf.linspace(-1.0, 1.0, dim_val)
        #     lin = tf.expand_dims(lin, 1 - dim)
        #     lin = tf.expand_dims(lin, 0)
        #     lin = tf.expand_dims(lin, 3)
        #     m = tf.reduce_max(input, [1, 2], keep_dims=True)
        #     e = tf.exp(input - m) + 1e-5
        #     val = tf.reduce_sum(e * lin, [1, 2]) / (tf.reduce_sum(e, [1, 2]))
        #     vals.append(tf.expand_dims(val, 2))
        #
        # return tf.reshape(tf.concat(2, vals), [-1, input.get_shape()[-1].value * 2])

        # import ipdb; ipdb.set_trace()

        # input.get_shape()
        # exp / tf.reduce_sum(exp, reduction_indices=[1, 2], keep_dims=True)
        # import ipdb;
        # ipdb.set_trace()
        # spatial softmax?

        # for dim in range(2):
        #     val = obs.get_shape()[dim + 1].value
        #     lin = tf.linspace(-1.0, 1.0, val)
        #     lin = tf.expand_dims(lin, 1 - dim)
        #     lin = tf.expand_dims(lin, 0)
        #     lin = tf.expand_dims(lin, 3)
        #     m = tf.reduce_max(e, [1, 2], keep_dims=True)
        #     e = tf.exp(e - m) + 1e-3
        #     val = tf.reduce_sum(e * lin, [1, 2]) / (tf.reduce_sum(e, [1, 2]))


class DropoutLayer(Layer):
    def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return input
        else:
            # Using theano constant to prevent upcasting
            # one = T.constant(1)

            retain_prob = 1. - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            return tf.nn.dropout(input, keep_prob=retain_prob)

    def get_output_shape_for(self, input_shape):
        return input_shape


# TODO: add Conv3DLayer

class FlattenLayer(Layer):
    """
    A layer that flattens its input. The leading ``outdim-1`` dimensions of
    the output will have the same shape as the input. The remaining dimensions
    are collapsed into the last dimension.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    outdim : int
        The number of dimensions in the output.
    See Also
    --------
    flatten  : Shortcut
    """

    def __init__(self, incoming, outdim=2, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)
        self.outdim = outdim

        if outdim < 1:
            raise ValueError('Dim must be >0, was %i', outdim)

    def get_output_shape_for(self, input_shape):
        to_flatten = input_shape[self.outdim - 1:]

        if any(s is None for s in to_flatten):
            flattened = None
        else:
            flattened = int(np.prod(to_flatten))

        return input_shape[:self.outdim - 1] + (flattened,)

    def get_output_for(self, input, **kwargs):
        # total_entries = tf.reduce_prod(tf.shape(input))
        pre_shape = tf.shape(input)[:self.outdim - 1]
        to_flatten = tf.reduce_prod(tf.shape(input)[self.outdim - 1:])
        return tf.reshape(input, tf.concat(0, [pre_shape, tf.pack([to_flatten])]))


flatten = FlattenLayer  # shortcut


class ReshapeLayer(Layer):
    def __init__(self, incoming, shape, **kwargs):
        super(ReshapeLayer, self).__init__(incoming, **kwargs)
        shape = tuple(shape)
        for s in shape:
            if isinstance(s, int):
                if s == 0 or s < - 1:
                    raise ValueError("`shape` integers must be positive or -1")
            elif isinstance(s, list):
                if len(s) != 1 or not isinstance(s[0], int) or s[0] < 0:
                    raise ValueError("`shape` input references must be "
                                     "single-element lists of int >= 0")
            elif isinstance(s, (tf.Tensor, tf.Variable)):  # T.TensorVariable):
                raise NotImplementedError
                # if s.ndim != 0:
                #     raise ValueError(
                #         "A symbolic variable in a shape specification must be "
                #         "a scalar, but had %i dimensions" % s.ndim)
            else:
                raise ValueError("`shape` must be a tuple of int and/or [int]")
        if sum(s == -1 for s in shape) > 1:
            raise ValueError("`shape` cannot contain multiple -1")
        self.shape = shape
        # try computing the output shape once as a sanity check
        self.get_output_shape_for(self.input_shape)

    def get_output_shape_for(self, input_shape, **kwargs):
        # Initialize output shape from shape specification
        output_shape = list(self.shape)
        # First, replace all `[i]` with the corresponding input dimension, and
        # mask parts of the shapes thus becoming irrelevant for -1 inference
        masked_input_shape = list(input_shape)
        masked_output_shape = list(output_shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                if o[0] >= len(input_shape):
                    raise ValueError("specification contains [%d], but input "
                                     "shape has %d dimensions only" %
                                     (o[0], len(input_shape)))
                output_shape[dim] = input_shape[o[0]]
                masked_output_shape[dim] = input_shape[o[0]]
                if (input_shape[o[0]] is None) \
                        and (masked_input_shape[o[0]] is None):
                    # first time we copied this unknown input size: mask
                    # it, we have a 1:1 correspondence between out[dim] and
                    # in[o[0]] and can ignore it for -1 inference even if
                    # it is unknown.
                    masked_input_shape[o[0]] = 1
                    masked_output_shape[dim] = 1
        # Secondly, replace all symbolic shapes with `None`, as we cannot
        # infer their size here.
        for dim, o in enumerate(output_shape):
            if isinstance(o, (tf.Tensor, tf.Variable)):  # T.TensorVariable):
                raise NotImplementedError
                # output_shape[dim] = None
                # masked_output_shape[dim] = None
        # From the shapes, compute the sizes of the input and output tensor
        input_size = (None if any(x is None for x in masked_input_shape)
                      else np.prod(masked_input_shape))
        output_size = (None if any(x is None for x in masked_output_shape)
                       else np.prod(masked_output_shape))
        del masked_input_shape, masked_output_shape
        # Finally, infer value for -1 if needed
        if -1 in output_shape:
            dim = output_shape.index(-1)
            if (input_size is None) or (output_size is None):
                output_shape[dim] = None
                output_size = None
            else:
                output_size *= -1
                output_shape[dim] = input_size // output_size
                output_size *= output_shape[dim]
        # Sanity check
        if (input_size is not None) and (output_size is not None) \
                and (input_size != output_size):
            raise ValueError("%s cannot be reshaped to specification %s. "
                             "The total size mismatches." %
                             (input_shape, self.shape))
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        # Replace all `[i]` with the corresponding input dimension
        output_shape = list(self.shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                output_shape[dim] = tf.shape(input)[o[0]]
        # Everything else is handled by Theano
        return tf.reshape(input, tf.pack(output_shape))


reshape = ReshapeLayer  # shortcut


class SliceLayer(Layer):
    def __init__(self, incoming, indices, axis=-1, **kwargs):
        super(SliceLayer, self).__init__(incoming, **kwargs)
        self.slice = indices
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        if isinstance(self.slice, int):
            del output_shape[self.axis]
        elif input_shape[self.axis] is not None:
            output_shape[self.axis] = len(
                list(range(*self.slice.indices(input_shape[self.axis]))))
        else:
            output_shape[self.axis] = None
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        axis = self.axis
        ndims = input.get_shape().ndims
        if axis < 0:
            axis += ndims
        if isinstance(self.slice, int) and self.slice < 0:
            return tf.reverse(input, [False] * self.axis + [True] + [False] * (ndims - axis - 1))[
                (slice(None),) * axis + (-1 - self.slice,) + (slice(None),) * (ndims - axis - 1)
                ]
        # import ipdb; ipdb.set_trace()
        return input[(slice(None),) * axis + (self.slice,) + (slice(None),) * (ndims - axis - 1)]


class DimshuffleLayer(Layer):
    def __init__(self, incoming, pattern, **kwargs):
        super(DimshuffleLayer, self).__init__(incoming, **kwargs)

        # Sanity check the pattern
        used_dims = set()
        for p in pattern:
            if isinstance(p, int):
                # Dimension p
                if p in used_dims:
                    raise ValueError("pattern contains dimension {0} more "
                                     "than once".format(p))
                used_dims.add(p)
            elif p == 'x':
                # Broadcast
                pass
            else:
                raise ValueError("pattern should only contain dimension"
                                 "indices or 'x', not {0}".format(p))

        self.pattern = pattern

        # try computing the output shape once as a sanity check
        self.get_output_shape_for(self.input_shape)

    def get_output_shape_for(self, input_shape):
        # Build output shape while keeping track of the dimensions that we are
        # attempting to collapse, so we can ensure that they are broadcastable
        output_shape = []
        dims_used = [False] * len(input_shape)
        for p in self.pattern:
            if isinstance(p, int):
                if p < 0 or p >= len(input_shape):
                    raise ValueError("pattern contains {0}, but input shape "
                                     "has {1} dimensions "
                                     "only".format(p, len(input_shape)))
                # Dimension p
                o = input_shape[p]
                dims_used[p] = True
            elif p == 'x':
                # Broadcast; will be of size 1
                o = 1
            output_shape.append(o)

        for i, (dim_size, used) in enumerate(zip(input_shape, dims_used)):
            if not used and dim_size != 1 and dim_size is not None:
                raise ValueError(
                    "pattern attempted to collapse dimension "
                    "{0} of size {1}; dimensions with size != 1/None are not"
                    "broadcastable and cannot be "
                    "collapsed".format(i, dim_size))

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return tf.transpose(input, self.pattern)


dimshuffle = DimshuffleLayer  # shortcut


def apply_ln(layer):
    def _normalize(x, prefix):
        EPS = 1e-5
        dim = x.get_shape()[-1].value

        bias_name = prefix + "_ln/bias"
        scale_name = prefix + "_ln/scale"

        if bias_name not in layer.norm_params:
            layer.norm_params[bias_name] = layer.add_param(
                tf.zeros_initializer, (dim,), name=bias_name, regularizable=False)
        if scale_name not in layer.norm_params:
            layer.norm_params[scale_name] = layer.add_param(
                tf.ones_initializer, (dim,), name=scale_name)

        bias = layer.norm_params[bias_name]
        scale = layer.norm_params[scale_name]
        mean, var = tf.nn.moments(x, axes=[1], keep_dims=True)
        x_normed = (x - mean) / tf.sqrt(var + EPS)
        return x_normed * scale + bias

    return _normalize


class GRULayer(Layer):
    """
    A gated recurrent unit implements the following update mechanism:
    Reset gate:        r(t) = f_r(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
    Update gate:       u(t) = f_u(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
    Cell gate:         c(t) = f_c(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
    New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u_t * c(t)
    Note that the reset, update, and cell vectors must have the same dimension as the hidden state
    """

    def __init__(self, incoming, num_units, hidden_nonlinearity,
                 gate_nonlinearity=tf.nn.sigmoid, W_x_init=XavierUniformInitializer(), W_h_init=OrthogonalInitializer(),
                 b_init=tf.zeros_initializer, hidden_init=tf.zeros_initializer, hidden_init_trainable=False,
                 layer_normalization=False, **kwargs):

        if hidden_nonlinearity is None:
            hidden_nonlinearity = tf.identity

        if gate_nonlinearity is None:
            gate_nonlinearity = tf.identity

        super(GRULayer, self).__init__(incoming, **kwargs)

        input_shape = self.input_shape[2:]

        input_dim = np.prod(input_shape)

        self.layer_normalization = layer_normalization

        # Weights for the initial hidden state
        self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                 regularizable=False)
        # Weights for the reset gate
        self.W_xr = self.add_param(W_x_init, (input_dim, num_units), name="W_xr")
        self.W_hr = self.add_param(W_h_init, (num_units, num_units), name="W_hr")
        self.b_r = self.add_param(b_init, (num_units,), name="b_r", regularizable=False)
        # Weights for the update gate
        self.W_xu = self.add_param(W_x_init, (input_dim, num_units), name="W_xu")
        self.W_hu = self.add_param(W_h_init, (num_units, num_units), name="W_hu")
        self.b_u = self.add_param(b_init, (num_units,), name="b_u", regularizable=False)
        # Weights for the cell gate
        self.W_xc = self.add_param(W_x_init, (input_dim, num_units), name="W_xc")
        self.W_hc = self.add_param(W_h_init, (num_units, num_units), name="W_hc")
        self.b_c = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)

        self.W_x_ruc = tf.concat(1, [self.W_xr, self.W_xu, self.W_xc])
        self.W_h_ruc = tf.concat(1, [self.W_hr, self.W_hu, self.W_hc])
        self.W_x_ru = tf.concat(1, [self.W_xr, self.W_xu])
        self.W_h_ru = tf.concat(1, [self.W_hr, self.W_hu])
        self.b_ruc = tf.concat(0, [self.b_r, self.b_u, self.b_c])

        self.gate_nonlinearity = gate_nonlinearity
        self.num_units = num_units
        self.nonlinearity = hidden_nonlinearity
        self.norm_params = dict()

        # pre-run the step method to initialize the normalization parameters
        h_dummy = tf.placeholder(dtype=tf.float32, shape=(None, num_units), name="h_dummy")
        x_dummy = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="x_dummy")
        self.step(h_dummy, x_dummy)

    def step(self, hprev, x):
        if self.layer_normalization:
            ln = apply_ln(self)
            x_ru = ln(tf.matmul(x, self.W_x_ru), "x_ru")
            h_ru = ln(tf.matmul(hprev, self.W_h_ru), "h_ru")
            x_r, x_u = tf.split(split_dim=1, num_split=2, value=x_ru)
            h_r, h_u = tf.split(split_dim=1, num_split=2, value=h_ru)
            x_c = ln(tf.matmul(x, self.W_xc), "x_c")
            h_c = ln(tf.matmul(hprev, self.W_hc), "h_c")
            r = self.gate_nonlinearity(x_r + h_r)
            u = self.gate_nonlinearity(x_u + h_u)
            c = self.nonlinearity(x_c + r * h_c)
            h = (1 - u) * hprev + u * c
            return h
        else:
            xb_ruc = tf.matmul(x, self.W_x_ruc) + tf.reshape(self.b_ruc, (1, -1))
            h_ruc = tf.matmul(hprev, self.W_h_ruc)
            xb_r, xb_u, xb_c = tf.split(split_dim=1, num_split=3, value=xb_ruc)
            h_r, h_u, h_c = tf.split(split_dim=1, num_split=3, value=h_ruc)
            r = self.gate_nonlinearity(xb_r + h_r)
            u = self.gate_nonlinearity(xb_u + h_u)
            c = self.nonlinearity(xb_c + r * h_c)
            h = (1 - u) * hprev + u * c
            return h

    def get_step_layer(self, l_in, l_prev_hidden, name=None):
        return GRUStepLayer(incomings=[l_in, l_prev_hidden], recurrent_layer=self, name=name)

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_output_for(self, input, **kwargs):
        input_shape = tf.shape(input)
        n_batches = input_shape[0]
        n_steps = input_shape[1]
        input = tf.reshape(input, tf.pack([n_batches, n_steps, -1]))
        if 'recurrent_state' in kwargs and self in kwargs['recurrent_state']:
            h0s = kwargs['recurrent_state'][self]
        else:
            h0s = tf.tile(
                tf.reshape(self.h0, (1, self.num_units)),
                (n_batches, 1)
            )
        # flatten extra dimensions
        shuffled_input = tf.transpose(input, (1, 0, 2))
        hs = tf.scan(
            self.step,
            elems=shuffled_input,
            initializer=h0s
        )
        shuffled_hs = tf.transpose(hs, (1, 0, 2))
        if 'recurrent_state_output' in kwargs:
            kwargs['recurrent_state_output'][self] = shuffled_hs
        return shuffled_hs


class GRUStepLayer(MergeLayer):
    def __init__(self, incomings, recurrent_layer, **kwargs):
        super(GRUStepLayer, self).__init__(incomings, **kwargs)
        self._gru_layer = recurrent_layer

    def get_params(self, **tags):
        return self._gru_layer.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0][0]
        return n_batch, self._gru_layer.num_units

    def get_output_for(self, inputs, **kwargs):
        x, hprev = inputs
        n_batch = tf.shape(x)[0]
        x = tf.reshape(x, tf.pack([n_batch, -1]))
        x.set_shape((None, self.input_shapes[0][1]))
        return self._gru_layer.step(hprev, x)


class TfGRULayer(Layer):
    """
    Use TensorFlow's built-in GRU implementation
    """

    def __init__(self, incoming, num_units, hidden_nonlinearity, horizon=None, hidden_init_trainable=False,
                 **kwargs):
        assert len(incoming.output_shape) == 3
        input_dim = incoming.shape[2]
        gru = tf.nn.rnn_cell.GRUCell(num_units=num_units, activation=hidden_nonlinearity)
        self.num_units = num_units
        self.horizon = horizon
        self.gru = gru
        self.hidden_nonlinearity = hidden_nonlinearity
        Layer.__init__(self, incoming=incoming, **kwargs)
        # dummy input variable
        input_dummy = tf.placeholder(tf.float32, (None, input_dim), "input_dummy")
        hidden_dummy = tf.placeholder(tf.float32, (None, num_units), "hidden_dummy")

        with tf.variable_scope(self.name) as vs:
            gru(input_dummy, hidden_dummy, scope=vs)
            vs.reuse_variables()
            self.scope = vs
            all_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            trainable_vars = [v for v in tf.trainable_variables() if v.name.startswith(vs.name)]

        for var in trainable_vars:
            self.add_param(spec=var, shape=None, name=None, trainable=True)
        for var in set(all_vars) - set(trainable_vars):
            self.add_param(spec=var, shape=None, name=None, trainable=False)
        self.h0 = self.add_param(tf.zeros_initializer, (num_units,), name="h0", trainable=hidden_init_trainable,
                                 regularizable=False)

    def step(self, hprev, x):
        return self.gru(x, hprev, scope=self.scope)[1]

    def get_output_for(self, input, **kwargs):
        input_shape = tf.shape(input)
        n_batches = input_shape[0]
        state = tf.tile(
            tf.reshape(self.h0, (1, self.num_units)),
            (n_batches, 1)
        )
        state.set_shape((None, self.num_units))
        if self.horizon is not None:
            outputs = []
            for idx in range(self.horizon):
                output, state = self.gru(input[:, idx, :], state, scope=self.scope)  # self.name)
                outputs.append(tf.expand_dims(output, 1))
            outputs = tf.concat(1, outputs)
            return outputs
        else:
            n_steps = input_shape[1]
            input = tf.reshape(input, tf.pack([n_batches, n_steps, -1]))
            # flatten extra dimensions
            shuffled_input = tf.transpose(input, (1, 0, 2))
            shuffled_input.set_shape((None, None, self.input_shape[-1]))
            hs = tf.scan(
                self.step,
                elems=shuffled_input,
                initializer=state
            )
            shuffled_hs = tf.transpose(hs, (1, 0, 2))
            return shuffled_hs

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_step_layer(self, l_in, l_prev_hidden, name=None):
        return GRUStepLayer(incomings=[l_in, l_prev_hidden], recurrent_layer=self, name=name)


class PseudoLSTMLayer(Layer):
    """
    A Pseudo LSTM unit implements the following update mechanism:

    Incoming gate:     i(t) = σ(W_hi @ h(t-1)) + W_xi @ x(t) + b_i)
    Forget gate:       f(t) = σ(W_hf @ h(t-1)) + W_xf @ x(t) + b_f)
    Out gate:          o(t) = σ(W_ho @ h(t-1)) + W_xo @ x(t) + b_o)
    New cell gate:     c_new(t) = ϕ(W_hc @ (o(t) * h(t-1)) + W_xc @ x(t) + b_c)
    Cell gate:         c(t) = f(t) * c(t-1) + i(t) * c_new(t)
    Hidden state:      h(t) = ϕ(c(t))
    Output:            out  = h(t)

    If gate_squash_inputs is set to True, we have the following updates instead:

    Out gate:          o(t) = σ(W_ho @ h(t-1)) + W_xo @ x(t) + b_o)
    Incoming gate:     i(t) = σ(W_hi @ (o(t) * h(t-1)) + W_xi @ x(t) + b_i)
    Forget gate:       f(t) = σ(W_hf @ (o(t) * h(t-1)) + W_xf @ x(t) + b_f)
    New cell gate:     c_new(t) = ϕ(W_hc @ (o(t) * h(t-1)) + W_xc @ x(t) + b_c)
    Cell state:        c(t) = f(t) * c(t-1) + i(t) * c_new(t)
    Hidden state:      h(t) = ϕ(c(t))
    Output:            out  = h(t)

    Note that the incoming, forget, cell, and out vectors must have the same dimension as the hidden state

    The notation is slightly different from
    http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html: here we introduce the cell
    gate and swap its role with the hidden state, so that the output is the same as the hidden state (and we can use
    this as a drop-in replacement for LSTMLayer).
    """

    def __init__(self, incoming, num_units, hidden_nonlinearity=tf.tanh,
                 gate_nonlinearity=tf.nn.sigmoid, W_x_init=XavierUniformInitializer(), W_h_init=OrthogonalInitializer(),
                 forget_bias=1.0, b_init=tf.zeros_initializer, hidden_init=tf.zeros_initializer,
                 hidden_init_trainable=False, cell_init=tf.zeros_initializer, cell_init_trainable=False,
                 gate_squash_inputs=False, layer_normalization=False, **kwargs):

        if hidden_nonlinearity is None:
            hidden_nonlinearity = tf.identity

        if gate_nonlinearity is None:
            gate_nonlinearity = tf.identity

        super(PseudoLSTMLayer, self).__init__(incoming, **kwargs)

        self.layer_normalization = layer_normalization

        input_shape = self.input_shape[2:]

        input_dim = np.prod(input_shape)
        # Weights for the initial hidden state (this is actually not used, since the initial hidden state is
        # determined by the initial cell state via h0 = self.nonlinearity(c0)). It is here merely for
        # interface convenience
        self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                 regularizable=False)
        # Weights for the initial cell state
        self.c0 = self.add_param(cell_init, (num_units,), name="c0", trainable=cell_init_trainable,
                                 regularizable=False)
        # Weights for the incoming gate
        self.W_xi = self.add_param(W_x_init, (input_dim, num_units), name="W_xi")
        self.W_hi = self.add_param(W_h_init, (num_units, num_units), name="W_hi")
        self.b_i = self.add_param(b_init, (num_units,), name="b_i", regularizable=False)
        # Weights for the forget gate
        self.W_xf = self.add_param(W_x_init, (input_dim, num_units), name="W_xf")
        self.W_hf = self.add_param(W_h_init, (num_units, num_units), name="W_hf")
        self.b_f = self.add_param(b_init, (num_units,), name="b_f", regularizable=False)
        # Weights for the out gate
        self.W_xo = self.add_param(W_x_init, (input_dim, num_units), name="W_xo")
        self.W_ho = self.add_param(W_h_init, (num_units, num_units), name="W_ho")
        self.b_o = self.add_param(b_init, (num_units,), name="b_o", regularizable=False)
        # Weights for the cell gate
        self.W_xc = self.add_param(W_x_init, (input_dim, num_units), name="W_xc")
        self.W_hc = self.add_param(W_h_init, (num_units, num_units), name="W_hc")
        self.b_c = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)

        self.gate_nonlinearity = gate_nonlinearity
        self.num_units = num_units
        self.nonlinearity = hidden_nonlinearity
        self.forget_bias = forget_bias
        self.gate_squash_inputs = gate_squash_inputs

        self.W_x_ifo = tf.concat(1, [self.W_xi, self.W_xf, self.W_xo])
        self.W_h_ifo = tf.concat(1, [self.W_hi, self.W_hf, self.W_ho])

        self.W_x_if = tf.concat(1, [self.W_xi, self.W_xf])
        self.W_h_if = tf.concat(1, [self.W_hi, self.W_hf])

        self.norm_params = dict()

    def step(self, hcprev, x):
        hprev = hcprev[:, :self.num_units]
        cprev = hcprev[:, self.num_units:]

        if self.layer_normalization:
            ln = apply_ln(self)
        else:
            ln = lambda x, *args: x

        if self.gate_squash_inputs:
            """
                Out gate:          o(t) = σ(W_ho @ h(t-1)) + W_xo @ x(t) + b_o)
                Incoming gate:     i(t) = σ(W_hi @ (o(t) * h(t-1)) + W_xi @ x(t) + b_i)
                Forget gate:       f(t) = σ(W_hf @ (o(t) * h(t-1)) + W_xf @ x(t) + b_f)
                New cell gate:     c_new(t) = ϕ(W_hc @ (o(t) * h(t-1)) + W_xc @ x(t) + b_c)
                Cell state:        c(t) = f(t) * c(t-1) + i(t) * c_new(t)
                Hidden state:      h(t) = ϕ(c(t))
                Output:            out  = h(t)
            """

            o = self.nonlinearity(
                ln(tf.matmul(hprev, self.W_ho), "h_o") +
                ln(tf.matmul(x, self.W_xo), "x_o") + self.b_o
            )

            x_if = ln(tf.matmul(x, self.W_x_if), "x_if")
            h_if = ln(tf.matmul(o * hprev, self.W_h_if), "h_if")

            x_i, x_f = tf.split(split_dim=1, num_split=2, value=x_if)
            h_i, h_f = tf.split(split_dim=1, num_split=2, value=h_if)

            i = self.gate_nonlinearity(x_i + h_i + self.b_i)
            f = self.gate_nonlinearity(x_f + h_f + self.b_f + self.forget_bias)
            c_new = self.nonlinearity(
                ln(tf.matmul(o * hprev, self.W_hc), "h_c") +
                ln(tf.matmul(x, self.W_xc), "x_c") +
                self.b_c
            )
            c = f * cprev + i * c_new
            h = self.nonlinearity(ln(c, "c"))
            return tf.concat(1, [h, c])
        else:
            """
                Incoming gate:     i(t) = σ(W_hi @ h(t-1)) + W_xi @ x(t) + b_i)
                Forget gate:       f(t) = σ(W_hf @ h(t-1)) + W_xf @ x(t) + b_f)
                Out gate:          o(t) = σ(W_ho @ h(t-1)) + W_xo @ x(t) + b_o)
                New cell gate:     c_new(t) = ϕ(W_hc @ (o(t) * h(t-1)) + W_xc @ x(t) + b_c)
                Cell gate:         c(t) = f(t) * c(t-1) + i(t) * c_new(t)
                Hidden state:      h(t) = ϕ(c(t))
                Output:            out  = h(t)
            """

            x_ifo = ln(tf.matmul(x, self.W_x_ifo), "x_ifo")
            h_ifo = ln(tf.matmul(hprev, self.W_h_ifo), "h_ifo")

            x_i, x_f, x_o = tf.split(split_dim=1, num_split=3, value=x_ifo)
            h_i, h_f, h_o = tf.split(split_dim=1, num_split=3, value=h_ifo)

            i = self.gate_nonlinearity(x_i + h_i + self.b_i)
            f = self.gate_nonlinearity(x_f + h_f + self.b_f + self.forget_bias)
            o = self.gate_nonlinearity(x_o + h_o + self.b_o)
            c_new = self.nonlinearity(
                ln(tf.matmul(o * hprev, self.W_hc), "h_c") +
                ln(tf.matmul(x, self.W_xc), "x_c") +
                self.b_c
            )
            c = f * cprev + i * c_new
            h = self.nonlinearity(ln(c, "c"))
            return tf.concat(1, [h, c])

    def get_step_layer(self, l_in, l_prev_state, name=None):
        return LSTMStepLayer(incomings=[l_in, l_prev_state], recurrent_layer=self, name=name)

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_output_for(self, input, **kwargs):
        input_shape = tf.shape(input)
        n_batches = input_shape[0]
        n_steps = input_shape[1]
        input = tf.reshape(input, tf.pack([n_batches, n_steps, -1]))
        c0s = tf.tile(
            tf.reshape(self.c0, (1, self.num_units)),
            (n_batches, 1)
        )
        h0s = self.nonlinearity(c0s)
        # flatten extra dimensions
        shuffled_input = tf.transpose(input, (1, 0, 2))
        hcs = tf.scan(
            self.step,
            elems=shuffled_input,
            initializer=tf.concat(1, [h0s, c0s])
        )
        shuffled_hcs = tf.transpose(hcs, (1, 0, 2))
        shuffled_hs = shuffled_hcs[:, :, :self.num_units]
        shuffled_cs = shuffled_hcs[:, :, self.num_units:]
        return shuffled_hs


class LSTMLayer(Layer):
    """
    A LSTM unit implements the following update mechanism:

    Incoming gate:     i(t) = f_i(x(t) @ W_xi + h(t-1) @ W_hi + w_ci * c(t-1) + b_i)
    Forget gate:       f(t) = f_f(x(t) @ W_xf + h(t-1) @ W_hf + w_cf * c(t-1) + b_f)
    Cell gate:         c(t) = f(t) * c(t - 1) + i(t) * f_c(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
    Out gate:          o(t) = f_o(x(t) @ W_xo + h(t-1) W_ho + w_co * c(t) + b_o)
    New hidden state:  h(t) = o(t) * f_h(c(t))

    Note that the incoming, forget, cell, and out vectors must have the same dimension as the hidden state
    """

    def __init__(self, incoming, num_units, hidden_nonlinearity=tf.tanh,
                 gate_nonlinearity=tf.nn.sigmoid, W_x_init=XavierUniformInitializer(), W_h_init=OrthogonalInitializer(),
                 forget_bias=1.0, use_peepholes=False, w_init=tf.random_normal_initializer(stddev=0.1),
                 b_init=tf.zeros_initializer, hidden_init=tf.zeros_initializer, hidden_init_trainable=False,
                 cell_init=tf.zeros_initializer, cell_init_trainable=False, layer_normalization=False,
                 **kwargs):

        if hidden_nonlinearity is None:
            hidden_nonlinearity = tf.identity

        if gate_nonlinearity is None:
            gate_nonlinearity = tf.identity

        super(LSTMLayer, self).__init__(incoming, **kwargs)

        self.layer_normalization = layer_normalization

        input_shape = self.input_shape[2:]

        input_dim = np.prod(input_shape)
        # Weights for the initial hidden state
        self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                 regularizable=False)
        # Weights for the initial cell state
        self.c0 = self.add_param(cell_init, (num_units,), name="c0", trainable=cell_init_trainable,
                                 regularizable=False)
        # Weights for the incoming gate
        self.W_xi = self.add_param(W_x_init, (input_dim, num_units), name="W_xi")
        self.W_hi = self.add_param(W_h_init, (num_units, num_units), name="W_hi")
        if use_peepholes:
            self.w_ci = self.add_param(w_init, (num_units,), name="w_ci")
        self.b_i = self.add_param(b_init, (num_units,), name="b_i", regularizable=False)
        # Weights for the forget gate
        self.W_xf = self.add_param(W_x_init, (input_dim, num_units), name="W_xf")
        self.W_hf = self.add_param(W_h_init, (num_units, num_units), name="W_hf")
        if use_peepholes:
            self.w_cf = self.add_param(w_init, (num_units,), name="w_cf")
        self.b_f = self.add_param(b_init, (num_units,), name="b_f", regularizable=False)
        # Weights for the cell gate
        self.W_xc = self.add_param(W_x_init, (input_dim, num_units), name="W_xc")
        self.W_hc = self.add_param(W_h_init, (num_units, num_units), name="W_hc")
        self.b_c = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)
        # Weights for the reset gate
        self.W_xr = self.add_param(W_x_init, (input_dim, num_units), name="W_xr")
        self.W_hr = self.add_param(W_h_init, (num_units, num_units), name="W_hr")
        self.b_r = self.add_param(b_init, (num_units,), name="b_r", regularizable=False)
        # Weights for the out gate
        self.W_xo = self.add_param(W_x_init, (input_dim, num_units), name="W_xo")
        self.W_ho = self.add_param(W_h_init, (num_units, num_units), name="W_ho")
        if use_peepholes:
            self.w_co = self.add_param(w_init, (num_units,), name="w_co")
        self.b_o = self.add_param(b_init, (num_units,), name="b_o", regularizable=False)
        self.gate_nonlinearity = gate_nonlinearity
        self.num_units = num_units
        self.nonlinearity = hidden_nonlinearity
        self.forget_bias = forget_bias
        self.use_peepholes = use_peepholes

        self.W_x_ifco = tf.concat(1, [self.W_xi, self.W_xf, self.W_xc, self.W_xo])
        self.W_h_ifco = tf.concat(1, [self.W_hi, self.W_hf, self.W_hc, self.W_ho])

        if use_peepholes:
            self.w_c_ifo = tf.concat(0, [self.w_ci, self.w_cf, self.w_co])

        self.norm_params = dict()

    def step(self, hcprev, x):
        """
            Incoming gate:     i(t) = f_i(x(t) @ W_xi + h(t-1) @ W_hi + w_ci * c(t-1) + b_i)
            Forget gate:       f(t) = f_f(x(t) @ W_xf + h(t-1) @ W_hf + w_cf * c(t-1) + b_f)
            Cell gate:         c(t) = f(t) * c(t - 1) + i(t) * f_c(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
            Out gate:          o(t) = f_o(x(t) @ W_xo + h(t-1) W_ho + w_co * c(t) + b_o)
            New hidden state:  h(t) = o(t) * f_h(c(t))
        """

        hprev = hcprev[:, :self.num_units]
        cprev = hcprev[:, self.num_units:]

        if self.layer_normalization:
            ln = apply_ln(self)
        else:
            ln = lambda x, *args: x

        x_ifco = ln(tf.matmul(x, self.W_x_ifco), "x_ifco")
        h_ifco = ln(tf.matmul(hprev, self.W_h_ifco), "h_ifco")
        x_i, x_f, x_c, x_o = tf.split(split_dim=1, num_split=4, value=x_ifco)
        h_i, h_f, h_c, h_o = tf.split(split_dim=1, num_split=4, value=h_ifco)

        if self.use_peepholes:
            i = self.gate_nonlinearity(x_i + h_i + self.w_ci * cprev + self.b_i)
            f = self.gate_nonlinearity(x_f + h_f + self.w_cf * cprev + self.b_f + self.forget_bias)

            o = self.gate_nonlinearity(x_o + h_o + self.w_co * cprev + self.b_o)
        else:
            i = self.gate_nonlinearity(x_i + h_i + self.b_i)
            f = self.gate_nonlinearity(x_f + h_f + self.b_f + self.forget_bias)
            c = f * cprev + i * self.nonlinearity(x_c + h_c + self.b_c)
            o = self.gate_nonlinearity(x_o + h_o + self.b_o)

        h = o * self.nonlinearity(ln(c, "c"))

        return tf.concat(1, [h, c])

    def get_step_layer(self, l_in, l_prev_state, name=None):
        return LSTMStepLayer(incomings=[l_in, l_prev_state], recurrent_layer=self, name=name)

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_output_for(self, input, **kwargs):
        input_shape = tf.shape(input)
        n_batches = input_shape[0]
        n_steps = input_shape[1]
        input = tf.reshape(input, tf.pack([n_batches, n_steps, -1]))
        h0s = tf.tile(
            tf.reshape(self.h0, (1, self.num_units)),
            (n_batches, 1)
        )
        c0s = tf.tile(
            tf.reshape(self.c0, (1, self.num_units)),
            (n_batches, 1)
        )
        # flatten extra dimensions
        shuffled_input = tf.transpose(input, (1, 0, 2))
        hcs = tf.scan(
            self.step,
            elems=shuffled_input,
            initializer=tf.concat(1, [h0s, c0s])
        )
        shuffled_hcs = tf.transpose(hcs, (1, 0, 2))
        shuffled_hs = shuffled_hcs[:, :, :self.num_units]
        shuffled_cs = shuffled_hcs[:, :, self.num_units:]
        if 'recurrent_state_output' in kwargs:
            kwargs['recurrent_state_output'][self] = shuffled_hcs
        return shuffled_hs


class LSTMStepLayer(MergeLayer):
    def __init__(self, incomings, recurrent_layer, **kwargs):
        super(LSTMStepLayer, self).__init__(incomings, **kwargs)
        self._recurrent_layer = recurrent_layer

    def get_params(self, **tags):
        return self._recurrent_layer.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0][0]
        return n_batch, 2 * self._recurrent_layer.num_units

    def get_output_for(self, inputs, **kwargs):
        x, hcprev = inputs
        n_batch = tf.shape(x)[0]
        x = tf.reshape(x, tf.pack([n_batch, -1]))
        hc = self._recurrent_layer.step(hcprev, x)
        return hc


class TfBasicLSTMLayer(Layer):
    """
    Use TensorFlow's built-in (basic) LSTM implementation
    """

    def __init__(self, incoming, num_units, hidden_nonlinearity, horizon=None, hidden_init_trainable=False,
                 forget_bias=1.0, use_peepholes=False, **kwargs):
        assert not use_peepholes, "Basic LSTM does not support peepholes!"
        assert len(incoming.output_shape) == 3
        input_dim = incoming.shape[2]
        lstm = tf.nn.rnn_cell.BasicLSTMCell(
            num_units=num_units,
            activation=hidden_nonlinearity,
            state_is_tuple=True,
            forget_bias=forget_bias
        )
        self.num_units = num_units
        self.horizon = horizon
        self.lstm = lstm
        self.hidden_nonlinearity = hidden_nonlinearity
        Layer.__init__(self, incoming=incoming, **kwargs)
        # dummy input variable
        input_dummy = tf.placeholder(tf.float32, (None, input_dim), "input_dummy")
        hidden_dummy = tf.placeholder(tf.float32, (None, num_units), "hidden_dummy")
        cell_dummy = tf.placeholder(tf.float32, (None, num_units), "cell_dummy")

        with tf.variable_scope(self.name) as vs:
            lstm(input_dummy, (cell_dummy, hidden_dummy), scope=vs)
            vs.reuse_variables()
            self.scope = vs
            all_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            trainable_vars = [v for v in tf.trainable_variables() if v.name.startswith(vs.name)]

        for var in trainable_vars:
            self.add_param(spec=var, shape=None, name=None, trainable=True)
        for var in set(all_vars) - set(trainable_vars):
            self.add_param(spec=var, shape=None, name=None, trainable=False)

        self.h0 = self.add_param(tf.zeros_initializer, (num_units,), name="h0", trainable=hidden_init_trainable,
                                 regularizable=False)
        self.c0 = self.add_param(tf.zeros_initializer, (num_units,), name="c0", trainable=hidden_init_trainable,
                                 regularizable=False)

    def step(self, hcprev, x):
        hprev = hcprev[:, :self.num_units]
        cprev = hcprev[:, self.num_units:]
        x.set_shape((None, self.input_shape[-1]))
        c, h = self.lstm(x, (cprev, hprev), scope=self.scope)[1]
        return tf.concat(1, [h, c])

    def get_output_for(self, input, **kwargs):
        input_shape = tf.shape(input)
        n_batches = input_shape[0]
        h0s = tf.tile(
            tf.reshape(self.h0, (1, self.num_units)),
            (n_batches, 1)
        )
        h0s.set_shape((None, self.num_units))
        c0s = tf.tile(
            tf.reshape(self.c0, (1, self.num_units)),
            (n_batches, 1)
        )
        c0s.set_shape((None, self.num_units))
        state = (c0s, h0s)
        if self.horizon is not None:
            outputs = []
            for idx in range(self.horizon):
                output, state = self.lstm(input[:, idx, :], state, scope=self.scope)  # self.name)
                outputs.append(tf.expand_dims(output, 1))
            outputs = tf.concat(1, outputs)
            return outputs
        else:
            n_steps = input_shape[1]
            input = tf.reshape(input, tf.pack([n_batches, n_steps, -1]))
            # flatten extra dimensions
            shuffled_input = tf.transpose(input, (1, 0, 2))
            shuffled_input.set_shape((None, None, self.input_shape[-1]))
            hcs = tf.scan(
                self.step,
                elems=shuffled_input,
                initializer=tf.concat(1, [h0s, c0s]),
            )
            shuffled_hcs = tf.transpose(hcs, (1, 0, 2))
            shuffled_hs = shuffled_hcs[:, :, :self.num_units]
            shuffled_cs = shuffled_hcs[:, :, self.num_units:]
            return shuffled_hs

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_step_layer(self, l_in, l_prev_state, name=None):
        return LSTMStepLayer(incomings=[l_in, l_prev_state], recurrent_layer=self, name=name)


def get_all_layers(layer, treat_as_input=None):
    """
    :type layer: Layer | list[Layer]
    :rtype: list[Layer]
    """
    # We perform a depth-first search. We add a layer to the result list only
    # after adding all its incoming layers (if any) or when detecting a cycle.
    # We use a LIFO stack to avoid ever running into recursion depth limits.
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()
    result = []

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        seen.update(treat_as_input)

    while queue:
        # Peek at the leftmost node in the queue.
        layer = queue[0]
        if layer is None:
            # Some node had an input_layer set to `None`. Just ignore it.
            queue.popleft()
        elif layer not in seen:
            # We haven't seen this node yet: Mark it and queue all incomings
            # to be processed first. If there are no incomings, the node will
            # be appended to the result list in the next iteration.
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle. In both cases, we remove the layer
            # from the queue and append it to the result list.
            queue.popleft()
            if layer not in done:
                result.append(layer)
                done.add(layer)

    return result


class NonlinearityLayer(Layer):
    def __init__(self, incoming, nonlinearity=tf.nn.relu, **kwargs):
        super(NonlinearityLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (tf.identity if nonlinearity is None
                             else nonlinearity)

    def get_output_for(self, input, **kwargs):
        return self.nonlinearity(input)

    def get_output_shape_for(self, input_shape):
        return input_shape


class BatchNormLayer(Layer):
    def __init__(self, incoming, center=True, scale=False, epsilon=0.001, decay=0.9,
                 beta=tf.zeros_initializer, gamma=tf.ones_initializer, moving_mean=tf.zeros_initializer,
                 moving_variance=tf.ones_initializer, **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.decay = decay

        input_shape = incoming.output_shape
        axis = list(range(len(input_shape) - 1))
        params_shape = input_shape[-1:]

        if center:
            self.beta = self.add_param(beta, shape=params_shape, name='beta', trainable=True, regularizable=False)
        else:
            self.beta = None
        if scale:
            self.gamma = self.add_param(gamma, shape=params_shape, name='gamma', trainable=True, regularizable=True)
        else:
            self.gamma = None

        self.moving_mean = self.add_param(moving_mean, shape=params_shape, name='moving_mean', trainable=False,
                                          regularizable=False)
        self.moving_variance = self.add_param(moving_variance, shape=params_shape, name='moving_variance',
                                              trainable=False, regularizable=False)
        self.axis = axis

    def get_output_for(self, input, phase='train', **kwargs):
        if phase == 'train':
            # Calculate the moments based on the individual batch.
            mean, variance = tf.nn.moments(input, self.axis, shift=self.moving_mean)
            # Update the moving_mean and moving_variance moments.
            update_moving_mean = moving_averages.assign_moving_average(
                self.moving_mean, mean, self.decay)
            update_moving_variance = moving_averages.assign_moving_average(
                self.moving_variance, variance, self.decay)
            # Make sure the updates are computed here.
            with tf.control_dependencies([update_moving_mean,
                                          update_moving_variance]):
                output = tf.nn.batch_normalization(
                    input, mean, variance, self.beta, self.gamma, self.epsilon)
        else:
            output = tf.nn.batch_normalization(
                input, self.moving_mean, self.moving_variance, self.beta, self.gamma, self.epsilon)
        output.set_shape(self.input_shape)
        return output

    def get_output_shape_for(self, input_shape):
        return input_shape


def batch_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    scale = True
    if nonlinearity is not None:
        layer.nonlinearity = tf.identity
        if nonlinearity is tf.nn.relu:
            scale = False
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn_name = (kwargs.pop('name', None) or
               (getattr(layer, 'name', None) and layer.name + '_bn'))
    layer = BatchNormLayer(layer, name=bn_name, scale=scale, **kwargs)
    if nonlinearity is not None:
        nonlin_name = bn_name and bn_name + '_nonlin'
        layer = NonlinearityLayer(layer, nonlinearity=nonlinearity, name=nonlin_name)
    return layer


class ElemwiseSumLayer(MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(ElemwiseSumLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        return functools.reduce(tf.add, inputs)

    def get_output_shape_for(self, input_shapes):
        assert len(set(input_shapes)) == 1
        return input_shapes[0]


def get_output(layer_or_layers, inputs=None, **kwargs):
    # track accepted kwargs used by get_output_for
    accepted_kwargs = {'deterministic'}
    # obtain topological ordering of all layers the output layer(s) depend on
    treat_as_input = list(inputs.keys()) if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-expression mapping from all input layers
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    # update layer-to-expression mapping from given input(s), if any
    if isinstance(inputs, dict):
        all_outputs.update((layer, tf.convert_to_tensor(expr))
                           for layer, expr in list(inputs.items()))
    elif inputs is not None:
        if len(all_outputs) > 1:
            raise ValueError("get_output() was called with a single input "
                             "expression on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input expressions instead.")
        for input_layer in all_outputs:
            all_outputs[input_layer] = tf.convert_to_tensor(inputs)
    # update layer-to-expression mapping by propagating the inputs
    for layer in all_layers:
        if layer not in all_outputs:
            try:
                if isinstance(layer, MergeLayer):
                    layer_inputs = [all_outputs[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    layer_inputs = all_outputs[layer.input_layer]
            except KeyError:
                # one of the input_layer attributes must have been `None`
                raise ValueError("get_output() was called without giving an "
                                 "input expression for the free-floating "
                                 "layer %r. Please call it with a dictionary "
                                 "mapping this layer to an input expression."
                                 % layer)
            all_outputs[layer] = layer.get_output_for(layer_inputs, **kwargs)
            try:
                names, _, _, defaults = getargspec(layer.get_output_for)
            except TypeError:
                # If introspection is not possible, skip it
                pass
            else:
                if defaults is not None:
                    accepted_kwargs |= set(names[-len(defaults):])
            accepted_kwargs |= set(layer.get_output_kwargs)
    unused_kwargs = set(kwargs.keys()) - accepted_kwargs
    if unused_kwargs:
        suggestions = []
        for kwarg in unused_kwargs:
            suggestion = get_close_matches(kwarg, accepted_kwargs)
            if suggestion:
                suggestions.append('%s (perhaps you meant %s)'
                                   % (kwarg, suggestion[0]))
            else:
                suggestions.append(kwarg)
        warn("get_output() was called with unused kwargs:\n\t%s"
             % "\n\t".join(suggestions))
    # return the output(s) of the requested layer(s) only
    try:
        return [all_outputs[layer] for layer in layer_or_layers]
    except TypeError:
        return all_outputs[layer_or_layers]


def unique(l):
    """Filters duplicates of iterable.
    Create a new list from l with duplicate entries removed,
    while preserving the original order.
    Parameters
    ----------
    l : iterable
        Input iterable to filter of duplicates.
    Returns
    -------
    list
        A list of elements of `l` without duplicates and in the same order.
    """
    new_list = []
    seen = set()
    for el in l:
        if el not in seen:
            new_list.append(el)
            seen.add(el)

    return new_list


def get_all_params(layer, **tags):
    """
    :type layer: Layer|list[Layer]
    """
    layers = get_all_layers(layer)
    params = chain.from_iterable(l.get_params(**tags) for l in layers)
    return unique(params)
