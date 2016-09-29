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

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a Theano tensor variable given the name and extra dimensions prepended
        :param name: name of the variable
        :param extra_dims: extra dimensions in the front
        :return: the created tensor variable
        """
        raise NotImplementedError
