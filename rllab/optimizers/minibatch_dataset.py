import numpy as np


class BatchDataset(object):

    def __init__(self, inputs, batch_size, extra_inputs=None):
        self._inputs = inputs
        if batch_size is None:
            batch_size = inputs[0].shape[0]
        self._batch_size = batch_size
        self._extra_inputs = extra_inputs
        self._ids = np.arange(self._inputs[0].shape[0])
        self.update()

    @property
    def number_batches(self):
        return int(np.ceil(self._inputs[0].shape[0] * 1.0 / self._batch_size))

    def iterate(self, update=True):
        for itr in xrange(self.number_batches):
            batch_start = itr * self._batch_size
            batch_end = (itr + 1) * self._batch_size
            batch_ids = self._ids[batch_start:batch_end]
            batch = [d[batch_ids] for d in self._inputs]
            if self._extra_inputs:
                yield batch + self._extra_inputs
            else:
                yield batch
        if update:
            self.update()

    def update(self):
        np.random.shuffle(self._ids)
