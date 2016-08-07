from __future__ import print_function
from __future__ import absolute_import
from rllab.misc import ext
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import scipy.optimize
import time


class LbfgsOptimizer(Serializable):
    """
    Performs unconstrained optimization via L-BFGS.
    """

    def __init__(self, name, max_opt_itr=20, callback=None):
        Serializable.quick_init(self, locals())
        self._name = name
        self._max_opt_itr = max_opt_itr
        self._opt_fun = None
        self._target = None
        self._callback = callback

    def update_opt(self, loss, target, inputs, extra_inputs=None, *args, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        def get_opt_output():
            flat_grad = tensor_utils.flatten_tensor_variables(tf.gradients(loss, target.get_params(trainable=True)))
            return [tf.cast(loss, tf.float64), tf.cast(flat_grad, tf.float64)]

        if extra_inputs is None:
            extra_inputs = list()

        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, loss),
            f_opt=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=get_opt_output(),
            )
        )

    def loss(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = list()
        return self._opt_fun["f_loss"](*(list(inputs) + list(extra_inputs)))

    def optimize(self, inputs, extra_inputs=None):
        f_opt = self._opt_fun["f_opt"]

        if extra_inputs is None:
            extra_inputs = list()

        def f_opt_wrapper(flat_params):
            self._target.set_param_values(flat_params, trainable=True)
            ret = f_opt(*inputs)
            return ret

        itr = [0]
        start_time = time.time()

        if self._callback:
            def opt_callback(params):
                loss = self._opt_fun["f_loss"](*(inputs + extra_inputs))
                elapsed = time.time() - start_time
                self._callback(dict(
                    loss=loss,
                    params=params,
                    itr=itr[0],
                    elapsed=elapsed,
                ))
                itr[0] += 1
        else:
            opt_callback = None

        scipy.optimize.fmin_l_bfgs_b(
            func=f_opt_wrapper, x0=self._target.get_param_values(trainable=True),
            maxiter=self._max_opt_itr, callback=opt_callback,
        )
