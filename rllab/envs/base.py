from .env_spec import EnvSpec
import collections
from cached_property import cached_property


class Env(object):
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        raise NotImplementedError

    @property
    def action_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        raise NotImplementedError

    # Helpers that derive from Spaces
    @property
    def action_dim(self):
        return self.action_space.flat_dim

    def render(self):
        pass

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @cached_property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    @property
    def horizon(self):
        """
        Horizon of the environment, if it has one
        """
        raise NotImplementedError


    def terminate(self):
        """
        Clean up operation,
        """
        pass

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass


_Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])


def Step(observation, reward, done, **kwargs):
    """
    Convenience method creating a namedtuple with the results of the
    environment.step method.
    Put extra diagnostic info in the kwargs
    """
    return _Step(observation, reward, done, kwargs)
