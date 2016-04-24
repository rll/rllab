import numpy as np
import pygame
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.envs.box2d.parser import find_body

from rllab.core.serializable import Serializable
from rllab.envs.box2d.parser.xml_box2d import _get_name
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class CarParkingEnv(Box2DEnv, Serializable):

    @autoargs.inherit(Box2DEnv.__init__)
    @autoargs.arg("random_start", type=bool,
                  help="Randomized starting position by uniforming sampling starting car angle"
                       "and position from a circle of radius 5")
    @autoargs.arg("random_start_range", type=float,
                  help="Defaulted to 1. which means possible angles are 1. * 2*pi")
    def __init__(self, *args, **kwargs):
        Serializable.__init__(self, *args, **kwargs)
        self.random_start = kwargs.pop("random_start", True)
        self.random_start_range = kwargs.pop("random_start_range", 1.)
        super(CarParkingEnv, self).__init__(
            self.model_path("car_parking.xml"),
            *args, **kwargs
        )
        self.goal = find_body(self.world, "goal")
        self.car = find_body(self.world, "car")
        self.wheels = [
            body for body in self.world.bodies if "wheel" in _get_name(body)]
        self.front_wheels = [
            body for body in self.wheels if "front" in _get_name(body)]
        self.max_deg = 30.
        self.goal_radius = 1.
        self.vel_thres = 1e-1
        self.start_radius = 5.

    @overrides
    def before_world_step(self, action):
        desired_angle = self.car.angle + action[-1] / 180 * np.pi
        for wheel in self.front_wheels:
            wheel.angle = desired_angle
            wheel.angularVelocity = 0  # kill angular velocity

        # kill all wheels' lateral speed
        for wheel in self.wheels:
            ortho = wheel.GetWorldVector((1, 0))
            lateral_speed = wheel.linearVelocity.dot(ortho) * ortho
            impulse = wheel.mass * -lateral_speed
            wheel.ApplyLinearImpulse(impulse, wheel.worldCenter, True)
            # also apply a tiny bit of fraction
            mag = wheel.linearVelocity.dot(wheel.linearVelocity)
            if mag != 0:
                wheel.ApplyLinearImpulse(
                    0.1 * wheel.mass * -wheel.linearVelocity / mag**0.5, wheel.worldCenter, True)

    @property
    @overrides
    def action_dim(self):
        return super(CarParkingEnv, self).action_dim + 1

    @property
    @overrides
    def action_bounds(self):
        lb, ub = super(CarParkingEnv, self).action_bounds
        return np.append(lb, -self.max_deg), np.append(ub, self.max_deg)

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        if self.random_start:
            pos_angle, car_angle = np.random.rand(
                2) * np.pi * 2 * self.random_start_range
            dis = (self.start_radius * np.cos(pos_angle),
                   self.start_radius * np.sin(pos_angle))
            for body in [self.car] + self.wheels:
                body.angle = car_angle
            for wheel in self.wheels:
                wheel.position = wheel.position - self.car.position + dis
            self.car.position = dis
            self.world.Step(
                self.extra_data.timeStep,
                self.extra_data.velocityIterations,
                self.extra_data.positionIterations
            )
        return self.get_current_obs()

    @overrides
    def compute_reward(self, action):
        yield
        not_done = not self.is_current_done()
        dist_to_goal = self.get_current_obs()[-3]
        yield - 1 * not_done - 2 * dist_to_goal

    @overrides
    def is_current_done(self):
        pos_satified = np.linalg.norm(self.car.position) <= self.goal_radius
        vel_satisfied = np.linalg.norm(
            self.car.linearVelocity) <= self.vel_thres
        return pos_satified and vel_satisfied

    @overrides
    def action_from_keys(self, keys):
        go = np.zeros(self.action_dim)
        if keys[pygame.K_LEFT]:
            go[-1] = self.max_deg
        if keys[pygame.K_RIGHT]:
            go[-1] = -self.max_deg
        if keys[pygame.K_UP]:
            go[0] = 10
        if keys[pygame.K_DOWN]:
            go[0] = -10
        return go

