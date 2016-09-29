from lasagne.layers import get_all_layers
from lasagne import utils


def get_full_output(layer_or_layers, inputs=None, **kwargs):
    """
    Computes the output of the network at one or more given layers.
    Optionally, you can define the input(s) to propagate through the network
    instead of using the input variable(s) associated with the network's
    input layer(s).

    Parameters
    ----------
    layer_or_layers : Layer or list
        the :class:`Layer` instance for which to compute the output
        expressions, or a list of :class:`Layer` instances.

    inputs : None, Theano expression, numpy array, or dict
        If None, uses the input variables associated with the
        :class:`InputLayer` instances.
        If a Theano expression, this defines the input for a single
        :class:`InputLayer` instance. Will throw a ValueError if there
        are multiple :class:`InputLayer` instances.
        If a numpy array, this will be wrapped as a Theano constant
        and used just like a Theano expression.
        If a dictionary, any :class:`Layer` instance (including the
        input layers) can be mapped to a Theano expression or numpy
        array to use instead of its regular output.

    Returns
    -------
    output : Theano expression or list
        the output of the given layer(s) for the given network input

    Notes
    -----
    Depending on your network architecture, `get_output([l1, l2])` may
    be crucially different from `[get_output(l1), get_output(l2)]`. Only
    the former ensures that the output expressions depend on the same
    intermediate expressions. For example, when `l1` and `l2` depend on
    a common dropout layer, the former will use the same dropout mask for
    both, while the latter will use two different dropout masks.
    """
    from lasagne.layers.input import InputLayer
    from lasagne.layers.base import MergeLayer
    # obtain topological ordering of all layers the output layer(s) depend on
    treat_as_input = list(inputs.keys()) if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-expression mapping from all input layers
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    extra_outputs = dict()
    # update layer-to-expression mapping from given input(s), if any
    if isinstance(inputs, dict):
        all_outputs.update((layer, utils.as_theano_expression(expr))
                           for layer, expr in list(inputs.items()))
    elif inputs is not None:
        # if len(all_outputs) > 1:
        #     raise ValueError("get_output() was called with a single input "
        #                      "expression on a network with multiple input "
        #                      "layers. Please call it with a dictionary of "
        #                      "input expressions instead.")
        for input_layer in all_outputs:
            all_outputs[input_layer] = utils.as_theano_expression(inputs)
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
            if hasattr(layer, "get_full_output_for"):
                output, extra = layer.get_full_output_for(layer_inputs, **kwargs)
                all_outputs[layer] = output
                extra_outputs[layer] = extra
            else:
                all_outputs[layer] = layer.get_output_for(
                    layer_inputs, **kwargs)
    # return the output(s) of the requested layer(s) only
    try:
        return [all_outputs[layer] for layer in layer_or_layers], extra_outputs
    except TypeError:
        return all_outputs[layer_or_layers], extra_outputs


def get_output(layer_or_layers, inputs=None, **kwargs):
    return get_full_output(layer_or_layers, inputs, **kwargs)[0]
