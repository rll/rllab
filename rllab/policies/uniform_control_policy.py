from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.policies.base import Policy
from rllab.misc.overrides import overrides


class UniformControlPolicy(Policy, Serializable):
    def __init__(
            self,
            env_spec,
    ):
        Serializable.quick_init(self, locals())
        super(UniformControlPolicy, self).__init__(env_spec=env_spec)

    @overrides
    def get_action(self, observation):
        return self.action_space.sample(), dict()

