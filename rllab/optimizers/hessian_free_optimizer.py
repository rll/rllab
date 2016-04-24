from rllab.misc.ext import compile_function, lazydict
from rllab.core.serializable import Serializable
from rllab.optimizers.hf import hf_optimizer
import time
from rllab.optimizers.minibatch_dataset import BatchDataset


class HessianFreeOptimizer(Serializable):
    """
    Performs unconstrained optimization via Hessian-Free Optimization
    """

    def __init__(self, max_opt_itr=20, batch_size=32, cg_batch_size=100, callback=None):
        Serializable.quick_init(self, locals())
        self._max_opt_itr = max_opt_itr
        self._opt_fun = None
        self._target = None
        self._batch_size = batch_size
        self._cg_batch_size = cg_batch_size
        self._hf_optimizer = None
        self._callback = callback

    def update_opt(self, loss, target, inputs, network_outputs, extra_inputs=None):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        if extra_inputs is None:
            extra_inputs = list()

        self._hf_optimizer = hf_optimizer(
            _p=target.get_params(trainable=True),
            inputs=(inputs + extra_inputs),
            s=network_outputs,
            costs=[loss],
        )

        self._opt_fun = lazydict(
            f_loss=lambda: compile_function(inputs + extra_inputs, loss),
        )

    def loss(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = list()
        return self._opt_fun["f_loss"](*(inputs + extra_inputs))

    def optimize(self, inputs, extra_inputs=None):

        if extra_inputs is None:
            extra_inputs = list()

#         import ipdb; ipdb.set_trace()
        dataset = BatchDataset(inputs=inputs, batch_size=self._batch_size, extra_inputs=extra_inputs)
        cg_dataset = BatchDataset(inputs=inputs, batch_size=self._cg_batch_size, extra_inputs=extra_inputs)

        itr = [0]
        start_time = time.time()

        if self._callback:
            def opt_callback():
                loss = self._opt_fun["f_loss"](*(inputs + extra_inputs))
                elapsed = time.time() - start_time
                self._callback(dict(
                    loss=loss,
                    params=self._target.get_param_values(trainable=True),
                    itr=itr[0],
                    elapsed=elapsed,
                ))
                itr[0] += 1
        else:
            opt_callback = None

        self._hf_optimizer.train(
            gradient_dataset=dataset,
            cg_dataset=cg_dataset,
            itr_callback=opt_callback,
            num_updates=self._max_opt_itr,
            preconditioner=True,
            verbose=True
        )
