import numpy as np
from rllab.envs.box2d.parser import find_body

from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class CartpoleEnv(Box2DEnv, Serializable):

    @autoargs.inherit(Box2DEnv.__init__)
    def __init__(self, *args, **kwargs):
        self.max_pole_angle = .2
        self.max_cart_pos = 2.4
        self.max_cart_speed = 4.
        self.max_pole_speed = 4.
        self.reset_range = 0.05
        super(CartpoleEnv, self).__init__(
            self.model_path("cartpole.xml.mako"),
            *args, **kwargs
        )
        self.cart = find_body(self.world, "cart")
        self.pole = find_body(self.world, "pole")
        Serializable.__init__(self, *args, **kwargs)

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        bounds = np.array([
            self.max_cart_pos,
            self.max_cart_speed,
            self.max_pole_angle,
            self.max_pole_speed
        ])
        low, high = -self.reset_range*bounds, self.reset_range*bounds
        xpos, xvel, apos, avel = np.random.uniform(low, high)
        self.cart.position = (xpos, self.cart.position[1])
        self.cart.linearVelocity = (xvel, self.cart.linearVelocity[1])
        self.pole.angle = apos
        self.pole.angularVelocity = avel
        return self.get_current_obs()

    @overrides
    def compute_reward(self, action):
        yield
        notdone = 1 - int(self.is_current_done())
        ucost = 1e-5*(action**2).sum()
        xcost = 1 - np.cos(self.pole.angle)
        yield notdone * 10 - notdone * xcost - notdone * ucost

    @overrides
    def is_current_done(self):
        return abs(self.cart.position[0]) > self.max_cart_pos or \
            abs(self.pole.angle) > self.max_pole_angle

