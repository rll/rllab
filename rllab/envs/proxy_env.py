from .base import Env


class ProxyEnv(Env):

    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self):
        return self._wrapped_env.reset()

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths):
        self._wrapped_env.log_diagnostics(paths)
