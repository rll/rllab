from rllab.misc import ext
from rllab.misc import logger
from rllab.core.serializable import Serializable
# from rllab.algo.first_order_method import parse_update_method
from rllab.optimizers.minibatch_dataset import BatchDataset
from collections import OrderedDict
import time
import lasagne.updates
import theano
import pyprind
from functools import partial


class FirstOrderOptimizer(Serializable):
    """
    Performs (stochastic) gradient descent, possibly using fancier methods like adam etc.
    """

    def __init__(
            self,
            update_method=lasagne.updates.adam,
            learning_rate=1e-3,
            max_epochs=1000,
            tolerance=1e-6,
            batch_size=32,
            callback=None,
            verbose=False,
            **kwargs):
        """

        :param max_epochs:
        :param tolerance:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will be used.
        :param callback:
        :param kwargs:
        :return:
        """
        Serializable.quick_init(self, locals())
        self._opt_fun = None
        self._target = None
        self._callback = callback
        update_method = partial(update_method, learning_rate=learning_rate)
        self._update_method = update_method
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._verbose = verbose

    def update_opt(self, loss, target, inputs, extra_inputs=None, gradients=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        if gradients is None:
            gradients = theano.grad(loss, target.get_params(trainable=True), disconnected_inputs='ignore')
        updates = self._update_method(gradients, target.get_params(trainable=True))
        updates = OrderedDict([(k, v.astype(k.dtype)) for k, v in updates.items()])

        if extra_inputs is None:
            extra_inputs = list()

        self._opt_fun = ext.lazydict(
            f_loss=lambda: ext.compile_function(inputs + extra_inputs, loss),
            f_opt=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss,
                updates=updates,
            )
        )

    def loss(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_loss"](*(tuple(inputs) + extra_inputs))

    def optimize_gen(self, inputs, extra_inputs=None, callback=None, yield_itr=None):

        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        f_opt = self._opt_fun["f_opt"]
        f_loss = self._opt_fun["f_loss"]

        if extra_inputs is None:
            extra_inputs = tuple()

        last_loss = f_loss(*(tuple(inputs) + extra_inputs))

        start_time = time.time()

        dataset = BatchDataset(
            inputs, self._batch_size,
            extra_inputs=extra_inputs
            #, randomized=self._randomized
        )

        itr = 0
        for epoch in pyprind.prog_bar(list(range(self._max_epochs))):
            for batch in dataset.iterate(update=True):
                f_opt(*batch)
                if yield_itr is not None and (itr % (yield_itr+1)) == 0:
                    yield
                itr += 1

            new_loss = f_loss(*(tuple(inputs) + extra_inputs))
            if self._verbose:
                logger.log("Epoch %d, loss %s" % (epoch, new_loss))

            if self._callback or callback:
                elapsed = time.time() - start_time
                callback_args = dict(
                    loss=new_loss,
                    params=self._target.get_param_values(trainable=True) if self._target else None,
                    itr=epoch,
                    elapsed=elapsed,
                )
                if self._callback:
                    self._callback(callback_args)
                if callback:
                    callback(**callback_args)

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss

    def optimize(self, inputs, **kwargs):
        for _ in self.optimize_gen(inputs, **kwargs):
            pass
