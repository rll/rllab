from rllab.core.parameterized import Parameterized


class Policy(Parameterized):
    def __init__(self, env_spec):
        Parameterized.__init__(self)
        self._env_spec = env_spec

    # Should be implemented by all policies

    def get_action(self, observation):
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def recurrent(self):
        """
        Indicates whether the policy is recurrent.
        :return:
        """
        return False

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def state_info_keys(self):
        """
        Return keys for the information related to the policy's state when taking an action.
        :return:
        """
        return list()

    def terminate(self):
        """
        Clean up operation
        """
        pass


class StochasticPolicy(Policy):

    @property
    def distribution(self):
        """
        :rtype Distribution
        """
        raise NotImplementedError

    def dist_info_sym(self, obs_var, state_info_vars):
        """
        Return the symbolic distribution information about the actions.
        :param obs_var: symbolic variable for observations
        :param state_info_vars: a dictionary whose values should contain information about the state of the policy at
        the time it received the observation
        :return:
        """
        raise NotImplementedError

    def dist_info(self, obs, state_infos):
        """
        Return the distribution information about the actions.
        :param obs_var: observation values
        :param state_info_vars: a dictionary whose values should contain information about the state of the policy at
        the time it received the observation
        :return:
        """
        raise NotImplementedError
