import tensorflow as tf

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized, suppress_params_loading


class Simple(Parameterized, Serializable):
    def __init__(self, name):
        Serializable.quick_init(self, locals())
        with tf.variable_scope(name):
            self.w = tf.get_variable("w", [10, 10])

    def get_params_internal(self, **tags):
        return [self.w]


class AllArgs(Serializable):
    def __init__(self, vararg, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.vararg = vararg
        self.args = args
        self.kwargs = kwargs


def test_serializable():
    with suppress_params_loading():
        obj = Simple(name="obj")
        obj1 = Serializable.clone(obj, name="obj1")
        assert obj.w.name.startswith('obj/')
        assert obj1.w.name.startswith('obj1/')

        obj2 = AllArgs(0, *(1,), **{'kwarg': 2})
        obj3 = Serializable.clone(obj2)
        assert obj3.vararg == 0
        assert len(obj3.args) == 1 and obj3.args[0] == 1
        assert len(obj3.kwargs) == 1 and obj3.kwargs['kwarg'] == 2


if __name__ == "__main__":
    test_serializable()
