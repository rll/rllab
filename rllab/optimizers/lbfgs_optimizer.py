from rllab.misc.ext import compile_function, lazydict, flatten_tensor_variables
from rllab.core.serializable import Serializable
import theano
import scipy.optimize
import time


class LbfgsOptimizer(Serializable):
    """
    Performs unconstrained optimization via L-BFGS.
    """

    def __init__(self, max_opt_itr=20, callback=None):
        Serializable.quick_init(self, locals())
        self._max_opt_itr = max_opt_itr
        self._opt_fun = None
        self._target = None
        self._callback = callback

    def update_opt(self, loss, target, inputs, extra_inputs=None, gradients=None, *args, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :param gradients: symbolic expressions for the gradients of trainable parameters of the target. By default
        this will be computed by calling theano.grad
        :return: No return value.
        """

        self._target = target

        def get_opt_output(gradients):
            if gradients is None:
                gradients = theano.grad(loss, target.get_params(trainable=True))
            flat_grad = flatten_tensor_variables(gradients)
            return [loss.astype('float64'), flat_grad.astype('float64')]

        if extra_inputs is None:
            extra_inputs = list()

        self._opt_fun = lazydict(
            f_loss=lambda: compile_function(inputs + extra_inputs, loss),
            f_opt=lambda: compile_function(
                inputs=inputs + extra_inputs,
                outputs=get_opt_output(gradients),
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
            return f_opt(*inputs)

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
