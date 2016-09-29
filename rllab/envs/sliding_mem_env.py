import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.spaces import Box


class SlidingMemEnv(ProxyEnv, Serializable):

    def __init__(
            self,
            env,
            n_steps=4,
            axis=0,
    ):
        super().__init__(env)
        Serializable.quick_init(self, locals())
        self.n_steps = n_steps
        self.axis = axis
        self.buffer = None

    def reset_buffer(self, new_):
        assert self.axis == 0
        self.buffer = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.buffer[0:] = new_

    def add_to_buffer(self, new_):
        assert self.axis == 0
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[:1] = new_

    @property
    def observation_space(self):
        origin = self._wrapped_env.observation_space
        return Box(
            *[
                np.repeat(b, self.n_steps, axis=self.axis)
                for b in origin.bounds
            ]
        )

    @overrides
    def reset(self):
        obs = self._wrapped_env.reset()
        self.reset_buffer(obs)
        return self.buffer

    @overrides
    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        self.add_to_buffer(next_obs)
        return Step(self.buffer, reward, done, **info)

