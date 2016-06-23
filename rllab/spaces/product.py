from rllab.spaces.base import Space
import numpy as np
from rllab.misc import ext


class Product(Space):

    def __init__(self, *components):
        if isinstance(components[0], (list, tuple)):
            assert len(components) == 1
            components = components[0]
        self._components = tuple(components)
        dtypes = [c.new_tensor_variable("tmp", extra_dims=0).dtype for c in components]
        self._common_dtype = np.core.numerictypes.find_common_type([], dtypes)

    def sample(self):
        return tuple(x.sample() for x in self._components)

    @property
    def components(self):
        return self._components

    def contains(self, x):
        return isinstance(x, tuple) and all(c.contains(xi) for c, xi in zip(self._components, x))

    def new_tensor_variable(self, name, extra_dims):
        return ext.new_tensor(
            name=name,
            ndim=extra_dims+1,
            dtype=self._common_dtype,
        )

    @property
    def flat_dim(self):
        return np.sum([c.flat_dim for c in self._components])

    def flatten(self, x):
        return np.concatenate([c.flatten(xi) for c, xi in zip(self._components, x)])

    def flatten_n(self, xs):
        xs_regrouped = [[x[i] for x in xs] for i in xrange(len(xs[0]))]
        flat_regrouped = [c.flatten_n(xi) for c, xi in zip(self.components, xs_regrouped)]
        return np.concatenate(flat_regrouped, axis=-1)

    def unflatten(self, x):
        dims = [c.flat_dim for c in self._components]
        flat_xs = np.split(x, np.cumsum(dims)[:-1])
        return tuple(c.unflatten(xi) for c, xi in zip(self._components, flat_xs))

    def unflatten_n(self, xs):
        dims = [c.flat_dim for c in self._components]
        flat_xs = np.split(xs, np.cumsum(dims)[:-1], axis=-1)
        unflat_xs = [c.unflatten_n(xi) for c, xi in zip(self.components, flat_xs)]
        unflat_xs_grouped = zip(*unflat_xs)
        return unflat_xs_grouped

    def __eq__(self, other):
        if not isinstance(other, Product):
            return False
        return tuple(self.components) == tuple(other.components)

    def __hash__(self):
        return hash(tuple(self.components))
