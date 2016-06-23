from rllab.algos.npo import NPO
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.misc import ext
from rllab.core.serializable import Serializable


class TNPG(NPO, Serializable):
    """
    Truncated Natural Policy Gradient.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(max_backtracks=1)
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TNPG, self).__init__(optimizer=optimizer, **kwargs)
