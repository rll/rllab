from rllab.algos.batch_polopt import BatchPolopt
from rllab.misc.overrides import overrides


class NOP(BatchPolopt):
    """
    NOP (no optimization performed) policy search algorithm
    """

    def __init__(
            self,
            **kwargs):
        super(NOP, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        pass

    @overrides
    def optimize_policy(self, itr, samples_data):
        pass

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict()
