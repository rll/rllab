from contextlib import contextmanager

from rllab.core.serializable import Serializable
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
import tensorflow as tf


load_params = True

@contextmanager
def suppress_params_loading():
    global load_params
    load_params = False
    yield
    load_params = True


class Parameterized(object):
    def __init__(self):
        self._cached_params = {}
        self._cached_param_dtypes = {}
        self._cached_param_shapes = {}
        self._cached_assign_ops = {}
        self._cached_assign_placeholders = {}

    def get_params_internal(self, **tags):
        """
        Internal method to be implemented which does not perform caching
        """
        raise NotImplementedError

    def get_params(self, **tags):
        """
        Get the list of parameters, filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(**tags)
        return self._cached_params[tag_tuple]

    def get_param_dtypes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def get_param_values(self, **tags):
        params = self.get_params(**tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, flattened_params, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(**tags))
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(**tags),
                self.get_param_dtypes(**tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, **tags):
        return unflatten_tensors(flattened_params, self.get_param_shapes(**tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.variables_initializer(self.get_params()))
            self.set_param_values(d["params"])


class JointParameterized(Parameterized):
    def __init__(self, components):
        super(JointParameterized, self).__init__()
        self.components = components

    def get_params_internal(self, **tags):
        params = [param for comp in self.components for param in comp.get_params_internal(**tags)]
        # only return unique parameters
        return sorted(set(params), key=hash)
