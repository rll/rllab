from rllab.misc import ext
from rllab.misc import logger
from rllab.core.serializable import Serializable
# from rllab.algo.first_order_method import parse_update_method
from rllab.optimizers.minibatch_dataset import BatchDataset
from collections import OrderedDict
import time
import lasagne.updates
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
        update_method = partial(lasagne.updates.adam, learning_rate=learning_rate)
        self._update_method = update_method
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._verbose = verbose

    def update_opt(self, loss, target, inputs, extra_inputs=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        updates = self._update_method(loss, target.get_params(trainable=True))
        updates = OrderedDict([(k, v.astype(k.dtype)) for k, v in updates.iteritems()])

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

    def optimize(self, inputs, extra_inputs=None, callback=None):

        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        f_opt = self._opt_fun["f_opt"]
        f_loss = self._opt_fun["f_loss"]

        if extra_inputs is None:
            extra_inputs = tuple()

        last_loss = f_loss(*(tuple(inputs) + extra_inputs))

        start_time = time.time()

        dataset = BatchDataset(inputs, self._batch_size, extra_inputs=extra_inputs)

        for epoch in xrange(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % epoch)
            for batch in dataset.iterate(update=True):
                f_opt(*batch)

            new_loss = f_loss(*(tuple(inputs) + extra_inputs))

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
