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


def test_serializable():
    with suppress_params_loading():
        obj = Simple(name="obj")
        obj1 = Serializable.clone(obj, name="obj1")
        assert obj.w.name.startswith('obj/')
        assert obj1.w.name.startswith('obj1/')


if __name__ == "__main__":
    test_serializable()
