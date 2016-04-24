import numpy as np
import pygame
from rllab.envs.box2d.parser import find_body

from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class MountainCarEnv(Box2DEnv, Serializable):

    @autoargs.inherit(Box2DEnv.__init__)
    @autoargs.arg("height_bonus_coeff", type=float,
                  help="Height bonus added to each step's reward")
    @autoargs.arg("goal_cart_pos", type=float,
                  help="Goal horizontal position")
    def __init__(self,
                 height_bonus=1.,
                 goal_cart_pos=0.6,
                 *args, **kwargs):
        super(MountainCarEnv, self).__init__(
            self.model_path("mountain_car.xml.mako"),
            *args, **kwargs
        )
        self.max_cart_pos = 2
        self.goal_cart_pos = goal_cart_pos
        self.height_bonus = height_bonus
        self.cart = find_body(self.world, "cart")
        Serializable.quick_init(self, locals())

    @overrides
    def compute_reward(self, action):
        yield
        yield (-1 + self.height_bonus * self.cart.position[1])

    @overrides
    def is_current_done(self):
        return self.cart.position[0] >= self.goal_cart_pos \
            or abs(self.cart.position[0]) >= self.max_cart_pos

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        bounds = np.array([
            [-1],
            [1],
        ])
        low, high = bounds
        xvel = np.random.uniform(low, high)
        self.cart.linearVelocity = (xvel, self.cart.linearVelocity[1])
        return self.get_current_obs()

    @overrides
    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-1])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+1])
        else:
            return np.asarray([0])

