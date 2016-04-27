from __future__ import print_function
from __future__ import absolute_import

import gym
import gym.envs
import gym.spaces
from gym.monitoring.monitor import capped_cubic_video_schedule
import os
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.misc import logger


def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError


class VideoSchedule(object):
    def __call__(self, count):
        return capped_cubic_video_schedule(count)


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


class GymEnv(Env, Serializable):
    def __init__(self, env_name, record_video=True, video_schedule=None):
        Serializable.quick_init(self, locals())
        env = gym.envs.make(env_name)
        self.env = env
        self.env_id = env.spec.id

        if logger.get_snapshot_dir() is None:
            logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = VideoSchedule()
            self.env.monitor.start(os.path.join(logger.get_snapshot_dir(), "gym_log"), video_schedule)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        self._action_space = convert_gym_space(env.action_space)
        self._horizon = env.spec.timestep_limit

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def terminate(self):
        if self.monitoring:
            self.env.monitor.close()
