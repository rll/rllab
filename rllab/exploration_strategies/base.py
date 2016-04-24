class ExplorationStrategy(object):
    def get_action(self, t, observation, policy, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass
