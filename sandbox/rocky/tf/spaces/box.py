from __future__ import print_function
from __future__ import absolute_import

from rllab.spaces.box import Box as TheanoBox
import tensorflow as tf


class Box(TheanoBox):
    def new_tensor_variable(self, name, extra_dims):
        return tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.flat_dim], name=name)
