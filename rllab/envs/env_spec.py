from rllab.core.serializable import Serializable
from rllab.spaces.base import Space


class EnvSpec(Serializable):

    def __init__(
            self,
            observation_space,
            action_space):
        """
        :type observation_space: Space
        :type action_space: Space
        """
        Serializable.quick_init(self, locals())
        self._observation_space = observation_space
        self._action_space = action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
