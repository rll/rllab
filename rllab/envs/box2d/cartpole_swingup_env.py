import numpy as np
import pygame
from rllab.envs.box2d.parser import find_body

from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


# Tornio, Matti, and Tapani Raiko. "Variational Bayesian approach for
# nonlinear identification and control." Proc. of the IFAC Workshop on
# Nonlinear Model Predictive Control for Fast Systems, NMPC FS06. 2006.
class CartpoleSwingupEnv(Box2DEnv, Serializable):

    @autoargs.inherit(Box2DEnv.__init__)
    def __init__(self, *args, **kwargs):
        super(CartpoleSwingupEnv, self).__init__(
            self.model_path("cartpole.xml.mako"),
            *args, **kwargs
        )
        self.max_cart_pos = 3
        self.max_reward_cart_pos = 3
        self.cart = find_body(self.world, "cart")
        self.pole = find_body(self.world, "pole")
        Serializable.__init__(self, *args, **kwargs)

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        bounds = np.array([
            [-1, -2, np.pi-1, -3],
            [1, 2, np.pi+1, 3],
        ])
        low, high = bounds
        xpos, xvel, apos, avel = np.random.uniform(low, high)
        self.cart.position = (xpos, self.cart.position[1])
        self.cart.linearVelocity = (xvel, self.cart.linearVelocity[1])
        self.pole.angle = apos
        self.pole.angularVelocity = avel
        return self.get_current_obs()

    @overrides
    def compute_reward(self, action):
        yield
        if self.is_current_done():
            yield -100
        else:
            if abs(self.cart.position[0]) > self.max_reward_cart_pos:
                yield -1
            else:
                yield np.cos(self.pole.angle)

    @overrides
    def is_current_done(self):
        return abs(self.cart.position[0]) > self.max_cart_pos

    @overrides
    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-10])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+10])
        else:
            return np.asarray([0])

