import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class NoisyObservationEnv(ProxyEnv, Serializable):

    @autoargs.arg('obs_noise', type=float,
                  help='Noise added to the observations (note: this makes the '
                       'problem non-Markovian!)')
    def __init__(self,
                 env,
                 obs_noise=1e-1,
                 ):
        super(NoisyObservationEnv, self).__init__(env)
        Serializable.quick_init(self, locals())
        self.obs_noise = obs_noise

    def get_obs_noise_scale_factor(self, obs):
        # return np.abs(obs)
        return np.ones_like(obs)

    def inject_obs_noise(self, obs):
        """
        Inject entry-wise noise to the observation. This should not change
        the dimension of the observation.
        """
        noise = self.get_obs_noise_scale_factor(obs) * self.obs_noise * \
            np.random.normal(size=obs.shape)
        return obs + noise

    def get_current_obs(self):
        return self.inject_obs_noise(self._wrapped_env.get_current_obs())

    @overrides
    def reset(self):
        obs = self._wrapped_env.reset()
        return self.inject_obs_noise(obs)

    @overrides
    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        return Step(self.inject_obs_noise(next_obs), reward, done, **info)


class DelayedActionEnv(ProxyEnv, Serializable):

    @autoargs.arg('action_delay', type=int,
                  help='Time steps before action is realized')
    def __init__(self,
                 env,
                 action_delay=3,
                 ):
        assert action_delay > 0, "Should not use this env transformer"
        super(DelayedActionEnv, self).__init__(env)
        Serializable.quick_init(self, locals())
        self.action_delay = action_delay
        self._queued_actions = None

    @overrides
    def reset(self):
        obs = self._wrapped_env.reset()
        self._queued_actions = np.zeros(self.action_delay * self.action_dim)
        return obs

    @overrides
    def step(self, action):
        queued_action = self._queued_actions[:self.action_dim]
        next_obs, reward, done, info = self._wrapped_env.step(queued_action)
        self._queued_actions = np.concatenate([
            self._queued_actions[self.action_dim:],
            action
        ])
        return Step(next_obs, reward, done, **info)

