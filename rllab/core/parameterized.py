from contextlib import contextmanager

from rllab.core.serializable import Serializable
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors

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

    def get_params_internal(self, **tags):
        """
        Internal method to be implemented which does not perform caching
        """
        raise NotImplementedError

    def get_params(self, **tags):  # adds the list to the _cached_params dict under the tuple key (one)
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
            self._cached_param_dtypes[tag_tuple] = \
                [param.get_value(borrow=True).dtype
                 for param in self.get_params(**tags)]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            self._cached_param_shapes[tag_tuple] = \
                [param.get_value(borrow=True).shape
                 for param in self.get_params(**tags)]
        return self._cached_param_shapes[tag_tuple]

    def get_param_values(self, **tags):
        return flatten_tensors(
            [param.get_value(borrow=True)
             for param in self.get_params(**tags)]
        )

    def set_param_values(self, flattened_params, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(**tags))
        for param, dtype, value in zip(
                self.get_params(**tags),
                self.get_param_dtypes(**tags),
                param_values):
            param.set_value(value.astype(dtype))
            if debug:
                print("setting value of %s" % param.name)

    def flat_to_params(self, flattened_params, **tags):
        return unflatten_tensors(flattened_params, self.get_param_shapes(**tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            self.set_param_values(d["params"])

