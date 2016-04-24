import numpy as np


class Space(object):
    """
    Provides a classification state spaces and action spaces,
    so you can write generic code that applies to any Environment.
    E.g. to choose a random action.
    """

    def sample(self, seed=0):
        """
        Uniformly randomly sample a random elemnt of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def flatten(self, x):
        raise NotImplementedError

    def unflatten(self, x):
        raise NotImplementedError

    def flatten_n(self, xs):
        raise NotImplementedError

    def unflatten_n(self, xs):
        raise NotImplementedError

    @property
    def flat_dim(self):
        """
        The dimension of the flattened vector of the tensor representation
        """
        raise NotImplementedError

    def new_tensor_variables(self, name, extra_dims):
        """
        Create one or a group of Theano tensor variables given the name and extra dimensions prepended
        :param name: name of the variable (or prefix, if the returned value is a group of variables)
        :param extra_dims: extra dimensions in the front
        :return: the created tensor variable(s)
        """
        raise NotImplementedError
