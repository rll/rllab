import os.path as osp

import mako.lookup
import mako.template
import numpy as np

from rllab import spaces
from rllab.envs.base import Env, Step
from rllab.envs.box2d.box2d_viewer import Box2DViewer

from rllab.envs.box2d.parser.xml_box2d import world_from_xml, find_body, \
    find_joint
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

BIG = 1e6
class Box2DEnv(Env):

    @autoargs.arg("frame_skip", type=int,
                  help="Number of frames to skip")
    @autoargs.arg('position_only', type=bool,
                  help='Whether to only provide (generalized) position as the '
                       'observation (i.e. no velocities etc.)')
    @autoargs.arg('obs_noise', type=float,
                  help='Noise added to the observations (note: this makes the '
                       'problem non-Markovian!)')
    @autoargs.arg('action_noise', type=float,
                  help='Noise added to the controls, which will be '
                       'proportional to the action bounds')
    def __init__(
            self, model_path, frame_skip=1, position_only=False,
            obs_noise=0.0, action_noise=0.0, template_string=None,
            template_args=None,
    ):
        self.full_model_path = model_path
        if template_string is None:
            if model_path.endswith(".mako"):
                with open(model_path) as template_file:
                    template = mako.template.Template(
                        template_file.read())
                template_string = template.render(
                    opts=template_args if template_args is not None else {},
                )
            else:
                with open(model_path, "r") as f:
                    template_string = f.read()
        world, extra_data = world_from_xml(template_string)
        self.world = world
        self.extra_data = extra_data
        self.initial_state = self._state
        self.viewer = None
        self.frame_skip = frame_skip
        self.timestep = self.extra_data.timeStep
        self.position_only = position_only
        self.obs_noise = obs_noise
        self.action_noise = action_noise
        self._action_bounds = None
        # cache the computation of position mask
        self._position_ids = None
        self._cached_obs = None
        self._cached_coms = {}

    def model_path(self, file_name):
        return osp.abspath(osp.join(osp.dirname(__file__),
                                    'models/%s' % file_name))

    def _set_state(self, state):
        splitted = np.array(state).reshape((-1, 6))
        for body, body_state in zip(self.world.bodies, splitted):
            xpos, ypos, apos, xvel, yvel, avel = body_state
            body.position = (xpos, ypos)
            body.angle = apos
            body.linearVelocity = (xvel, yvel)
            body.angularVelocity = avel

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        return self.get_current_obs()

    def _invalidate_state_caches(self):
        self._cached_obs = None
        self._cached_coms = {}

    @property
    def _state(self):
        s = []
        for body in self.world.bodies:
            s.append(np.concatenate([
                list(body.position),
                [body.angle],
                list(body.linearVelocity),
                [body.angularVelocity]
            ]))
        return np.concatenate(s)

    @property
    @overrides
    def action_space(self):
        lb = np.array([control.ctrllimit[0] for control in self.extra_data.controls])
        ub = np.array([control.ctrllimit[1] for control in self.extra_data.controls])
        return spaces.Box(lb, ub)

    @property
    @overrides
    def observation_space(self):
        if self.position_only:
            d = len(self._get_position_ids())
        else:
            d = len(self.extra_data.states)
        ub = BIG * np.ones(d)
        return spaces.Box(ub*-1, ub)

    @property
    def action_bounds(self):
        return self.action_space.bounds

    def forward_dynamics(self, action):
        if len(action) != self.action_dim:
            raise ValueError('incorrect action dimension: expected %d but got '
                             '%d' % (self.action_dim, len(action)))
        lb, ub = self.action_bounds
        action = np.clip(action, lb, ub)
        for ctrl, act in zip(self.extra_data.controls, action):
            if ctrl.typ == "force":
                for name in ctrl.bodies:
                    body = find_body(self.world, name)
                    direction = np.array(ctrl.direction)
                    direction = direction / np.linalg.norm(direction)
                    world_force = body.GetWorldVector(direction * act)
                    world_point = body.GetWorldPoint(ctrl.anchor)
                    body.ApplyForce(world_force, world_point, wake=True)
            elif ctrl.typ == "torque":
                assert ctrl.joint
                joint = find_joint(self.world, ctrl.joint)
                joint.motorEnabled = True
                # forces the maximum allowed torque to be taken
                if act > 0:
                    joint.motorSpeed = 1e5
                else:
                    joint.motorSpeed = -1e5
                joint.maxMotorTorque = abs(act)
            else:
                raise NotImplementedError
        self.before_world_step(action)
        self.world.Step(
            self.extra_data.timeStep,
            self.extra_data.velocityIterations,
            self.extra_data.positionIterations
        )

    def compute_reward(self, action):
        """
        The implementation of this method should have two parts, structured
        like the following:

        <perform calculations before stepping the world>
        yield
        reward = <perform calculations after stepping the world>
        yield reward
        """
        raise NotImplementedError

    @overrides
    def step(self, action):
        """
        Note: override this method with great care, as it post-processes the
        observations, etc.
        """
        reward_computer = self.compute_reward(action)
        # forward the state
        action = self._inject_action_noise(action)
        for _ in range(self.frame_skip):
            self.forward_dynamics(action)
        # notifies that we have stepped the world
        next(reward_computer)
        # actually get the reward
        reward = next(reward_computer)
        self._invalidate_state_caches()
        done = self.is_current_done()
        next_obs = self.get_current_obs()
        return Step(observation=next_obs, reward=reward, done=done)

    def _filter_position(self, obs):
        """
        Filter the observation to contain only position information.
        """
        return obs[self._get_position_ids()]

    def get_obs_noise_scale_factor(self, obs):
        return np.ones_like(obs)

    def _inject_obs_noise(self, obs):
        """
        Inject entry-wise noise to the observation. This should not change
        the dimension of the observation.
        """
        noise = self.get_obs_noise_scale_factor(obs) * self.obs_noise * \
            np.random.normal(size=obs.shape)
        return obs + noise

    def get_current_reward(
            self, state, xml_obs, action, next_state, next_xml_obs):
        raise NotImplementedError

    def is_current_done(self):
        raise NotImplementedError

    def _inject_action_noise(self, action):
        # generate action noise
        noise = self.action_noise * \
            np.random.normal(size=action.shape)
        # rescale the noise to make it proportional to the action bounds
        lb, ub = self.action_bounds
        noise = 0.5 * (ub - lb) * noise
        return action + noise

    def get_current_obs(self):
        """
        This method should not be overwritten.
        """
        raw_obs = self.get_raw_obs()
        noisy_obs = self._inject_obs_noise(raw_obs)
        if self.position_only:
            return self._filter_position(noisy_obs)
        return noisy_obs

    def _get_position_ids(self):
        if self._position_ids is None:
            self._position_ids = []
            for idx, state in enumerate(self.extra_data.states):
                if state.typ in ["xpos", "ypos", "apos", "dist", "angle"]:
                    self._position_ids.append(idx)
        return self._position_ids

    def get_raw_obs(self):
        """
        Return the unfiltered & noiseless observation. By default, it computes
        based on the declarations in the xml file.
        """
        if self._cached_obs is not None:
            return self._cached_obs
        obs = []
        for state in self.extra_data.states:
            new_obs = None
            if state.body:
                body = find_body(self.world, state.body)
                if state.local is not None:
                    l = state.local
                    position = body.GetWorldPoint(l)
                    linearVel = body.GetLinearVelocityFromLocalPoint(l)
                    # now I wish I could write angle = error "not supported"
                else:
                    position = body.position
                    linearVel = body.linearVelocity

                if state.to is not None:
                    to = find_body(self.world, state.to)

                if state.typ == "xpos":
                    new_obs = position[0]
                elif state.typ == "ypos":
                    new_obs = position[1]
                elif state.typ == "xvel":
                    new_obs = linearVel[0]
                elif state.typ == "yvel":
                    new_obs = linearVel[1]
                elif state.typ == "apos":
                    new_obs = body.angle
                elif state.typ == "avel":
                    new_obs = body.angularVelocity
                elif state.typ == "dist":
                    new_obs = np.linalg.norm(position - to.position)
                elif state.typ == "angle":
                    diff = to.position - position
                    abs_angle = np.arccos(
                        diff.dot((0, 1)) / np.linalg.norm(diff))
                    new_obs = body.angle + abs_angle
                else:
                    raise NotImplementedError
            elif state.joint:
                joint = find_joint(self.world, state.joint)
                if state.typ == "apos":
                    new_obs = joint.angle
                elif state.typ == "avel":
                    new_obs = joint.speed
                else:
                    raise NotImplementedError
            elif state.com:
                com_quant = self._compute_com_pos_vel(*state.com)
                if state.typ == "xpos":
                    new_obs = com_quant[0]
                elif state.typ == "ypos":
                    new_obs = com_quant[1]
                elif state.typ == "xvel":
                    new_obs = com_quant[2]
                elif state.typ == "yvel":
                    new_obs = com_quant[3]
                else:
                    print(state.typ)
                    # orientation and angular velocity of the whole body is not
                    # supported
                    raise NotImplementedError
            else:
                raise NotImplementedError

            if state.transform is not None:
                if state.transform == "id":
                    pass
                elif state.transform == "sin":
                    new_obs = np.sin(new_obs)
                elif state.transform == "cos":
                    new_obs = np.cos(new_obs)
                else:
                    raise NotImplementedError

            obs.append(new_obs)

        self._cached_obs = np.array(obs)
        return self._cached_obs

    def _compute_com_pos_vel(self, *com):
        com_key = ",".join(sorted(com))
        if com_key in self._cached_coms:
            return self._cached_coms[com_key]
        total_mass_quant = 0
        total_mass = 0
        for body_name in com:
            body = find_body(self.world, body_name)
            total_mass_quant += body.mass * np.array(
                list(body.worldCenter) + list(body.linearVelocity))
            total_mass += body.mass
        com_quant = total_mass_quant / total_mass
        self._cached_coms[com_key] = com_quant
        return com_quant

    def get_com_position(self, *com):
        return self._compute_com_pos_vel(*com)[:2]

    def get_com_velocity(self, *com):
        return self._compute_com_pos_vel(*com)[2:]

    @overrides
    def render(self, states=None, actions=None, pause=False):
        if not self.viewer:
            self.viewer = Box2DViewer(self.world)
        if states or actions or pause:
            raise NotImplementedError
        if not self.viewer:
            self.start_viewer()
        if self.viewer:
            self.viewer.loop_once()

    def before_world_step(self, action):
        pass

    def action_from_keys(self, keys):
        raise NotImplementedError

