from rllab.core.parameterized import Parameterized
from rllab.misc.overrides import overrides
import lasagne.layers as L


class LasagnePowered(Parameterized):
    def __init__(self, output_layers):
        self._output_layers = output_layers
        super(LasagnePowered, self).__init__()

    @property
    def output_layers(self):
        return self._output_layers

    @overrides
    def get_params_internal(self, **tags):  # this gives ALL the vars (not the params values)
        return L.get_all_params(  # this lasagne function also returns all var below the passed layers
            L.concat(self._output_layers),
            **tags
        )
