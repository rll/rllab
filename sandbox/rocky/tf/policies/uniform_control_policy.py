from sandbox.rocky.tf.policies.base import Policy
from rllab.core.serializable import Serializable


class UniformControlPolicy(Policy, Serializable):
    def __init__(
            self,
            env_spec,
    ):
        Serializable.quick_init(self, locals())
        super(UniformControlPolicy, self).__init__(env_spec=env_spec)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        return self.action_space.sample(), dict()

    def get_actions(self, observations):
        return self.action_space.sample_n(len(observations)), dict()

    def get_params_internal(self, **tags):
        return []
