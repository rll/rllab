from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import math
import tensorflow as tf
from collections import OrderedDict
from collections import deque
from itertools import chain
from inspect import getargspec
from difflib import get_close_matches
from warnings import warn


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
    def __init__(self, incoming, name, **kwargs):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming
        self.params = OrderedDict()
        self.name = name
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

    def add_param(self, spec, shape, name, **tags):
        with tf.variable_scope(self.name):
            tags['trainable'] = tags.get('trainable', True)
            tags['regularizable'] = tags.get('regularizable', True)
            param = create_param(spec, shape, name, **tags)
            self.params[param] = set(tag for tag, value in tags.items() if value)
            return param

    def get_params(self, **tags):
        result = list(self.params.keys())

        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - self.params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (self.params[param] & exclude)]

        return result


class InputLayer(Layer):
    def __init__(self, shape, name, input_var=None, **kwargs):
        super(InputLayer, self).__init__(shape, name, **kwargs)
        self.shape = shape
        if input_var is None:
            with tf.variable_scope(name):
                input_var = tf.placeholder(tf.float32, shape=shape, name="input")
        self.input_var = input_var

    @Layer.output_shape.getter
    def output_shape(self):
        return self.shape


class MergeLayer(Layer):
    def __init__(self, incomings, name, **kwargs):
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

    def __init__(self, incomings, name, axis=1, **kwargs):
        super(ConcatLayer, self).__init__(incomings, name, **kwargs)
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


def xavier_init(shape, dtype=tf.float32):
    if len(shape) == 2:
        n_inputs, n_outputs = shape
    else:
        receptive_field_size = np.prod(shape[:2])
        n_inputs = shape[-2] * receptive_field_size
        n_outputs = shape[-1] * receptive_field_size
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)


def he_init(shape, dtype=tf.float32):
    if len(shape) == 2:
        n_inputs, _ = shape
    else:
        receptive_field_size = np.prod(shape[:2])
        n_inputs = shape[-2] * receptive_field_size
    init_range = math.sqrt(1.0 / n_inputs)
    return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)


class ParamLayer(Layer):
    def __init__(self, incoming, name, num_units, param=tf.zeros_initializer,
                 trainable=True, **kwargs):
        super(ParamLayer, self).__init__(incoming, name, **kwargs)
        self.num_units = num_units
        with tf.variable_scope(name):
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
    def __init__(self, incoming, name, op,
                 shape_op=lambda x: x, extras=None, **kwargs):
        if extras is None:
            extras = []
        incomings = [incoming] + extras
        super(OpLayer, self).__init__(incomings, name, **kwargs)
        self.op = op
        self.shape_op = shape_op
        self.incomings = incomings

    def get_output_shape_for(self, input_shapes):
        return self.shape_op(*input_shapes)

    def get_output_for(self, inputs, **kwargs):
        return self.op(*inputs)


class DenseLayer(Layer):
    def __init__(self, incoming, name, num_units, nonlinearity=None, W=xavier_init, b=tf.zeros_initializer,
                 **kwargs):
        super(DenseLayer, self).__init__(incoming, name, **kwargs)
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
                 W=xavier_init, b=tf.zeros_initializer,
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
        # if pad == 'VALID':
        #     self.pad = as_tuple(0, n)
        # elif pad in ('full', 'same'):
        #     self.pad = pad
        # else:
        #     self.pad = as_tuple(pad, n, int)

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
                 W=xavier_init, b=tf.zeros_initializer,
                 nonlinearity=tf.nn.relu,
                 convolution=tf.nn.conv2d, **kwargs):
        super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, n=2, **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        conved = self.convolution(input, self.W, strides=(1,) + self.stride + (1,), padding=self.pad)
        return conved


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

    def __init__(self, incoming, name, outdim=2, **kwargs):
        super(FlattenLayer, self).__init__(incoming, name=name, **kwargs)
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
    def __init__(self, incoming, shape, name, **kwargs):
        super(ReshapeLayer, self).__init__(incoming, name=name, **kwargs)
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
    def __init__(self, incoming, name, indices, axis=-1, **kwargs):
        super(SliceLayer, self).__init__(incoming, name, **kwargs)
        self.slice = indices
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        if isinstance(self.slice, int):
            del output_shape[self.axis]
        elif input_shape[self.axis] is not None:
            output_shape[self.axis] = len(
                range(*self.slice.indices(input_shape[self.axis])))
        else:
            output_shape[self.axis] = None
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        axis = self.axis
        ndims = input.get_shape().ndims
        if axis < 0:
            axis += ndims
        return input[(slice(None),) * axis + (self.slice,) + (slice(None),) * (ndims - axis - 1)]


class DimshuffleLayer(Layer):
    def __init__(self, incoming, pattern, name, **kwargs):
        super(DimshuffleLayer, self).__init__(incoming, name=name, **kwargs)

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


class GRULayer(Layer):
    """
    A gated recurrent unit implements the following update mechanism:
    Reset gate:        r(t) = f_r(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
    Update gate:       u(t) = f_u(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
    Cell gate:         c(t) = f_c(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
    New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u_t * c(t)
    Note that the reset, update, and cell vectors must have the same dimension as the hidden state
    """

    def __init__(self, incoming, name, num_units, hidden_nonlinearity,
                 gate_nonlinearity=tf.nn.sigmoid, W_init=xavier_init, b_init=tf.zeros_initializer,
                 hidden_init=tf.zeros_initializer, hidden_init_trainable=True):

        if hidden_nonlinearity is None:
            hidden_nonlinearity = tf.identity

        if gate_nonlinearity is None:
            gate_nonlinearity = tf.identity

        super(GRULayer, self).__init__(incoming, name=name)

        with tf.variable_scope(name):

            input_shape = self.input_shape[2:]

            input_dim = np.prod(input_shape)
            # self._name = name
            # Weights for the initial hidden state
            self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                     regularizable=False)
            # Weights for the reset gate
            self.W_xr = self.add_param(W_init, (input_dim, num_units), name="W_xr")
            self.W_hr = self.add_param(W_init, (num_units, num_units), name="W_hr")
            self.b_r = self.add_param(b_init, (num_units,), name="b_r", regularizable=False)
            # Weights for the update gate
            self.W_xu = self.add_param(W_init, (input_dim, num_units), name="W_xu")
            self.W_hu = self.add_param(W_init, (num_units, num_units), name="W_hu")
            self.b_u = self.add_param(b_init, (num_units,), name="b_u", regularizable=False)
            # Weights for the cell gate
            self.W_xc = self.add_param(W_init, (input_dim, num_units), name="W_xc")
            self.W_hc = self.add_param(W_init, (num_units, num_units), name="W_hc")
            self.b_c = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)

            self.W_x_ruc = tf.concat(1, [self.W_xr, self.W_xu, self.W_xc])
            self.W_h_ruc = tf.concat(1, [self.W_hr, self.W_hu, self.W_hc])

            self.gate_nonlinearity = gate_nonlinearity
            self.num_units = num_units
            self.nonlinearity = hidden_nonlinearity

    def step(self, hprev, x):
        x_ruc = tf.matmul(x, self.W_x_ruc)
        h_ruc = tf.matmul(hprev, self.W_h_ruc)
        x_r, x_u, x_c = tf.split(split_dim=1, num_split=3, value=x_ruc)
        h_r, h_u, h_c = tf.split(split_dim=1, num_split=3, value=h_ruc)
        r = self.gate_nonlinearity(x_r + h_r + self.b_r)
        u = self.gate_nonlinearity(x_u + h_u + self.b_u)
        c = self.nonlinearity(x_c + r * h_c + self.b_c)
        h = (1 - u) * hprev + u * c
        return h

    def get_step_layer(self, l_in, l_prev_hidden, name):
        return GRUStepLayer(incomings=[l_in, l_prev_hidden], gru_layer=self, name=name)

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
        # flatten extra dimensions
        shuffled_input = tf.transpose(input, (1, 0, 2))
        hs = tf.scan(
            self.step,
            elems=shuffled_input,
            initializer=h0s
        )
        shuffled_hs = tf.transpose(hs, (1, 0, 2))
        return shuffled_hs


class GRUStepLayer(MergeLayer):
    def __init__(self, incomings, name, gru_layer):
        super(GRUStepLayer, self).__init__(incomings, name)
        self._gru_layer = gru_layer

    def get_params(self, **tags):
        return self._gru_layer.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0]
        return n_batch, self._gru_layer.num_units

    def get_output_for(self, inputs, **kwargs):
        x, hprev = inputs
        n_batch = tf.shape(x)[0]
        x = tf.reshape(x, tf.pack([n_batch, -1]))
        return self._gru_layer.step(hprev, x)


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

    def __init__(self, incoming, name, num_units, hidden_nonlinearity,
                 gate_nonlinearity=tf.nn.sigmoid, W_init=xavier_init, forget_bias=1.0,
                 use_peepholes=False,
                 w_init=tf.random_normal_initializer(stddev=0.1),
                 b_init=tf.zeros_initializer, hidden_init=tf.zeros_initializer, hidden_init_trainable=True,
                 cell_init=tf.zeros_initializer, cell_init_trainable=True):

        if hidden_nonlinearity is None:
            hidden_nonlinearity = tf.identity

        if gate_nonlinearity is None:
            gate_nonlinearity = tf.identity

        super(LSTMLayer, self).__init__(incoming, name=name)

        with tf.variable_scope(name):

            input_shape = self.input_shape[2:]

            input_dim = np.prod(input_shape)
            # Weights for the initial hidden state
            self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                     regularizable=False)
            # Weights for the initial cell state
            self.c0 = self.add_param(cell_init, (num_units,), name="c0", trainable=cell_init_trainable,
                                     regularizable=False)
            # Weights for the incoming gate
            self.W_xi = self.add_param(W_init, (input_dim, num_units), name="W_xi")
            self.W_hi = self.add_param(W_init, (num_units, num_units), name="W_hi")
            if use_peepholes:
                self.w_ci = self.add_param(w_init, (num_units,), name="w_ci")
            self.b_i = self.add_param(b_init, (num_units,), name="b_i", regularizable=False)
            # Weights for the forget gate
            self.W_xf = self.add_param(W_init, (input_dim, num_units), name="W_xf")
            self.W_hf = self.add_param(W_init, (num_units, num_units), name="W_hf")
            if use_peepholes:
                self.w_cf = self.add_param(w_init, (num_units,), name="w_cf")
            self.b_f = self.add_param(b_init, (num_units,), name="b_f", regularizable=False)
            # Weights for the cell gate
            self.W_xc = self.add_param(W_init, (input_dim, num_units), name="W_xc")
            self.W_hc = self.add_param(W_init, (num_units, num_units), name="W_hc")
            self.b_c = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)
            # Weights for the reset gate
            self.W_xr = self.add_param(W_init, (input_dim, num_units), name="W_xr")
            self.W_hr = self.add_param(W_init, (num_units, num_units), name="W_hr")
            self.b_r = self.add_param(b_init, (num_units,), name="b_r", regularizable=False)
            # Weights for the out gate
            self.W_xo = self.add_param(W_init, (input_dim, num_units), name="W_xo")
            self.W_ho = self.add_param(W_init, (num_units, num_units), name="W_ho")
            if use_peepholes:
                self.w_co = self.add_param(w_init, (num_units,), name="w_co")
            self.b_o = self.add_param(b_init, (num_units,), name="b_o", regularizable=False)
            self.gate_nonlinearity = gate_nonlinearity
            self.num_units = num_units
            self.nonlinearity = hidden_nonlinearity
            self.forget_bias = forget_bias
            self.use_peepholes = use_peepholes

            self.W_x_ifco = tf.concat(1, [self.W_xi, self.W_xf, self.W_xo, self.W_xi])
            self.W_h_ifco = tf.concat(1, [self.W_hi, self.W_hf, self.W_ho, self.W_hi])
            if use_peepholes:
                self.w_c_ifo = tf.concat(0, [self.w_ci, self.w_cf, self.w_co])

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

        x_ifco = tf.matmul(x, self.W_x_ifco)
        h_ifco = tf.matmul(hprev, self.W_h_ifco)
        x_i, x_f, x_c, x_o = tf.split(split_dim=1, num_split=4, value=x_ifco)
        h_i, h_f, h_c, h_o = tf.split(split_dim=1, num_split=4, value=h_ifco)

        if self.use_peepholes:
            i = self.gate_nonlinearity(x_i + h_i + self.w_ci * cprev + self.b_i)
            f = self.gate_nonlinearity(x_f + h_f + self.w_cf * cprev + self.b_f + self.forget_bias)
            c = f * cprev + i * self.nonlinearity(x_c + h_c + self.b_c)
            o = self.gate_nonlinearity(x_o + h_o + self.w_co * cprev + self.b_o)
        else:
            i = self.gate_nonlinearity(x_i + h_i + self.b_i)
            f = self.gate_nonlinearity(x_f + h_f + self.b_f + self.forget_bias)
            c = f * cprev + i * self.nonlinearity(x_c + h_c + self.b_c)
            o = self.gate_nonlinearity(x_o + h_o + self.b_o)

        h = o * self.nonlinearity(c)

        return tf.concat(1, [h, c])

    def get_step_layer(self, l_in, l_prev_hidden, l_prev_cell, name):
        return LSTMStepLayer(incomings=[l_in, l_prev_hidden, l_prev_cell], lstm_layer=self, name=name)

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
        return shuffled_hs


class LSTMStepLayer(MergeLayer):
    def __init__(self, incomings, name, lstm_layer):
        super(LSTMStepLayer, self).__init__(incomings, name)
        self._lstm_layer = lstm_layer

    def get_params(self, **tags):
        return self._lstm_layer.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0]
        return n_batch, 2 * self._lstm_layer.num_units

    def get_output_for(self, inputs, **kwargs):
        x, hprev, cprev = inputs
        n_batch = tf.shape(x)[0]
        x = tf.reshape(x, tf.pack([n_batch, -1]))
        hcprev = tf.concat(1, [hprev, cprev])
        hc = self._lstm_layer.step(hcprev, x)
        return hc


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


def get_output(layer_or_layers, inputs=None, **kwargs):
    # track accepted kwargs used by get_output_for
    accepted_kwargs = {'deterministic'}
    # obtain topological ordering of all layers the output layer(s) depend on
    treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-expression mapping from all input layers
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    # update layer-to-expression mapping from given input(s), if any
    if isinstance(inputs, dict):
        all_outputs.update((layer, tf.convert_to_tensor(expr))
                           for layer, expr in inputs.items())
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
