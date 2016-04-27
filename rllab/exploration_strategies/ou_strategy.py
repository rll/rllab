from rllab.misc.overrides import overrides
from rllab.misc.ext import AttrDict
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np
import numpy.random as nr


class OUStrategy(ExplorationStrategy, Serializable):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """

    def __init__(self, env_spec, mu=0, theta=0.15, sigma=0.3, **kwargs):
        assert isinstance(env_spec.action_space, Box)
        assert len(env_spec.action_space.shape) == 1
        Serializable.quick_init(self, locals())
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = env_spec.action_space
        self.state = np.ones(self.action_space.flat_dim) * self.mu
        self.reset()

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["state"] = self.state
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.state = d["state"]

    @overrides
    def reset(self):
        self.state = np.ones(self.action_space.flat_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    @overrides
    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        ou_state = self.evolve_state()
        return np.clip(action + ou_state, self.action_space.low, self.action_space.high)


if __name__ == "__main__":
    ou = OUStrategy(env_spec=AttrDict(action_space=Box(low=-1, high=1, shape=(1,))), mu=0, theta=0.15, sigma=0.3)
    states = []
    for i in range(1000):
        states.append(ou.evolve_state()[0])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
