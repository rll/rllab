from rllab.algos.vpg import VPG
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.core.serializable import Serializable


class ERWR(VPG, Serializable):
    """
    Episodic Reward Weighted Regression [1]_

    Notes
    -----
    This does not implement the original RwR [2]_ that deals with "immediate reward problems" since
    it doesn't find solutions that optimize for temporally delayed rewards.

    .. [1] Kober, Jens, and Jan R. Peters. "Policy search for motor primitives in robotics." Advances in neural information processing systems. 2009.
    .. [2] Peters, Jan, and Stefan Schaal. "Using reward-weighted regression for reinforcement learning of task space control." Approximate Dynamic Programming and Reinforcement Learning, 2007. ADPRL 2007. IEEE International Symposium on. IEEE, 2007.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            positive_adv=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = LbfgsOptimizer(**optimizer_args)
        super(ERWR, self).__init__(
            optimizer=optimizer,
            positive_adv=True if positive_adv is None else positive_adv,
            **kwargs
        )

