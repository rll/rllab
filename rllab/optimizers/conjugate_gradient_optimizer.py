from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
import theano.tensor as TT
import theano
import itertools
import numpy as np


class ConjugateGradientOptimizer(Serializable):
    """
    Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
    algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
    of the loss function.
    """

    def __init__(
            self,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=0.1,
            backtrack_ratio=0.8,
            max_backtracks=15,
            debug_nan=False):
        """

        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
        :param debug_nan: if set to True, NanGuard will be added to the compilation, and ipdb will be invoked when
        nan is detected
        :return:
        """
        Serializable.quick_init(self, locals())
        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._debug_nan = debug_nan

    def update_opt(self, loss, target, leq_constraint, inputs, extra_inputs=None, constraint_name="constraint", *args,
                   **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs, which could be subsampled if needed. It is assumed
        that the first dimension of these inputs should correspond to the number of data points
        :param extra_inputs: A list of symbolic variables as extra inputs which should not be subsampled
        :return: No return value.
        """

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        constraint_term, constraint_value = leq_constraint

        params = target.get_params(trainable=True)
        grads = theano.grad(loss, wrt=params)
        flat_grad = ext.flatten_tensor_variables(grads)

        constraint_grads = theano.grad(constraint_term, wrt=params)
        xs = tuple([ext.new_tensor_like("%s x" % p.name, p) for p in params])
        Hx_plain_splits = TT.grad(
            TT.sum([TT.sum(g * x) for g, x in itertools.izip(constraint_grads, xs)]),
            wrt=params,
        )
        Hx_plain = TT.concatenate([TT.flatten(s) for s in Hx_plain_splits])

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name


        if self._debug_nan:
            from theano.compile.nanguardmode import NanGuardMode
            mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        else:
            mode = None

        self._opt_fun = ext.lazydict(
            f_loss=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss,
                log_name="f_loss",
                mode=mode,
            ),
            f_grad=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
                log_name="f_grad",
                mode=mode,
            ),
            f_Hx_plain=lambda: ext.compile_function(
                inputs=inputs + extra_inputs + xs,
                outputs=Hx_plain,
                log_name="f_Hx_plain",
                mode=mode,
            ),
            f_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=constraint_term,
                log_name="constraint",
                mode=mode,
            ),
            f_loss_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=[loss, constraint_term],
                log_name="f_loss_constraint",
                mode=mode,
            ),
        )

    def loss(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_loss"](*(inputs + extra_inputs))

    def constraint_val(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_constraint"](*(inputs + extra_inputs))

    def optimize(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()

        if self._subsample_factor < 1:
            n_samples = len(inputs[0])
            inds = np.random.choice(
                n_samples, int(n_samples * self._subsample_factor), replace=False)
            subsample_inputs = tuple([x[inds] for x in inputs])
        else:
            subsample_inputs = inputs

        logger.log("computing loss before")
        loss_before = self._opt_fun["f_loss"](*(inputs + extra_inputs))
        logger.log("performing update")
        logger.log("computing descent direction")

        flat_g = self._opt_fun["f_grad"](*(inputs + extra_inputs))

        def Hx(x):
            xs = tuple(self._target.flat_to_params(x, trainable=True))
            #     rop = f_Hx_rop(*(inputs + xs))
            plain = self._opt_fun["f_Hx_plain"](*(subsample_inputs + extra_inputs + xs)) + self._reg_coeff * x
            # assert np.allclose(rop, plain)
            return plain
            # alternatively we can do finite difference on flat_grad

        descent_direction = krylov.cg(Hx, flat_g, cg_iters=self._cg_iters)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")

        prev_param = self._target.get_param_values(trainable=True)
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self._target.set_param_values(cur_param, trainable=True)
            loss, constraint_val = self._opt_fun["f_loss_constraint"](*(inputs + extra_inputs))
            if self._debug_nan and np.isnan(constraint_val):
                import ipdb; ipdb.set_trace()
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break
        logger.log("backtrack iters: %d" % n_iter)
        logger.log("computing loss after")
        logger.log("optimization finished")

