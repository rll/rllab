import math
import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
from ctypes import byref

import numpy as np
import theano

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Env, Step
from rllab.envs.mujoco.gather.embedded_viewer import EmbeddedViewer
from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.mujoco_py import MjViewer, MjModel, mjcore, mjlib, \
    mjextra, glfw

APPLE = 0
BOMB = 1


class GatherViewer(MjViewer):
    def __init__(self, env):
        self.env = env
        super(GatherViewer, self).__init__()
        green_ball_model = MjModel(osp.abspath(
            osp.join(
                MODEL_DIR, 'green_ball.xml'
            )
        ))
        self.green_ball_renderer = EmbeddedViewer()
        self.green_ball_model = green_ball_model
        self.green_ball_renderer.set_model(green_ball_model)
        red_ball_model = MjModel(osp.abspath(
            osp.join(
                MODEL_DIR, 'red_ball.xml'
            )
        ))
        self.red_ball_renderer = EmbeddedViewer()
        self.red_ball_model = red_ball_model
        self.red_ball_renderer.set_model(red_ball_model)

    def start(self):
        super(GatherViewer, self).start()
        self.green_ball_renderer.start(self.window)
        self.red_ball_renderer.start(self.window)

    def handle_mouse_move(self, window, xpos, ypos):
        super(GatherViewer, self).handle_mouse_move(window, xpos, ypos)
        self.green_ball_renderer.handle_mouse_move(window, xpos, ypos)
        self.red_ball_renderer.handle_mouse_move(window, xpos, ypos)

    def handle_scroll(self, window, x_offset, y_offset):
        super(GatherViewer, self).handle_scroll(window, x_offset, y_offset)
        self.green_ball_renderer.handle_scroll(window, x_offset, y_offset)
        self.red_ball_renderer.handle_scroll(window, x_offset, y_offset)

    def render(self):
        super(GatherViewer, self).render()
        tmpobjects = mjcore.MJVOBJECTS()
        mjlib.mjlib.mjv_makeObjects(byref(tmpobjects), 1000)
        for obj in self.env.objects:
            x, y, typ = obj
            # print x, y
            qpos = np.zeros_like(self.green_ball_model.data.qpos)
            qpos[0, 0] = x
            qpos[1, 0] = y
            if typ == APPLE:
                self.green_ball_model.data.qpos = qpos
                self.green_ball_model.forward()
                self.green_ball_renderer.render()
                mjextra.append_objects(
                    tmpobjects, self.green_ball_renderer.objects)
            else:
                self.red_ball_model.data.qpos = qpos
                self.red_ball_model.forward()
                self.red_ball_renderer.render()
                mjextra.append_objects(
                    tmpobjects, self.red_ball_renderer.objects)
        mjextra.append_objects(tmpobjects, self.objects)
        mjlib.mjlib.mjv_makeLights(
            self.model.ptr, self.data.ptr, byref(tmpobjects))
        mjlib.mjlib.mjr_render(0, self.get_rect(), byref(tmpobjects), byref(
            self.ropt), byref(self.cam.pose), byref(self.con))

        try:
            import OpenGL.GL as GL
        except:
            return

        def draw_rect(x, y, width, height):
            # start drawing a rectangle
            GL.glBegin(GL.GL_QUADS)
            # bottom left point
            GL.glVertex2f(x, y)
            # bottom right point
            GL.glVertex2f(x + width, y)
            # top right point
            GL.glVertex2f(x + width, y + height)
            # top left point
            GL.glVertex2f(x, y + height)
            GL.glEnd()

        def refresh2d(width, height):
            GL.glViewport(0, 0, width, height)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GL.glOrtho(0.0, width, 0.0, height, 0.0, 1.0)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

        GL.glLoadIdentity()
        width, height = glfw.get_framebuffer_size(self.window)
        refresh2d(width, height)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_BLEND)

        GL.glColor4f(0.0, 0.0, 0.0, 0.8)
        draw_rect(10, 10, 300, 100)

        apple_readings, bomb_readings = self.env.get_readings()
        for idx, reading in enumerate(apple_readings):
            if reading > 0:
                GL.glColor4f(0.0, 1.0, 0.0, reading)
                draw_rect(20 * (idx + 1), 10, 5, 50)
        for idx, reading in enumerate(bomb_readings):
            if reading > 0:
                GL.glColor4f(1.0, 0.0, 0.0, reading)
                draw_rect(20 * (idx + 1), 60, 5, 50)


class GatherEnv(Env, Serializable):
    MODEL_CLASS = None
    ORI_IND = None

    @autoargs.arg('n_apples', type=int,
                  help='Number of apples in each episode')
    @autoargs.arg('n_bombs', type=int,
                  help='Number of bombs in each episode')
    @autoargs.arg('activity_range', type=float,
                  help='The span for generating objects '
                       '(x, y in [-range, range])')
    @autoargs.arg('robot_object_spacing', type=float,
                  help='Number of objects in each episode')
    @autoargs.arg('catch_range', type=float,
                  help='Minimum distance range to catch an object')
    @autoargs.arg('n_bins', type=float,
                  help='Number of objects in each episode')
    @autoargs.arg('sensor_range', type=float,
                  help='Maximum sensor range (how far it can go)')
    @autoargs.arg('sensor_span', type=float,
                  help='Maximum sensor span (how wide it can span), in '
                       'radians')
    def __init__(
            self,
            n_apples=8,
            n_bombs=8,
            activity_range=6.,
            robot_object_spacing=2.,
            catch_range=1.,
            n_bins=10,
            sensor_range=6.,
            sensor_span=math.pi,
            *args, **kwargs
    ):
        self.n_apples = n_apples
        self.n_bombs = n_bombs
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.objects = []
        super(GatherEnv, self).__init__(*args, **kwargs)
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        xml_path = osp.join(MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")
        attrs = dict(
            type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3"
        )
        walldist = self.activity_range + 1
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall1",
                pos="0 -%d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall2",
                pos="0 %d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall3",
                pos="-%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall4",
                pos="%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path)
        # pylint: disable=not-callable
        inner_env = model_cls(*args, file_path=file_path, **kwargs)
        # pylint: enable=not-callable
        self.inner_env = inner_env
        Serializable.quick_init(self, locals())

    def reset(self):
        # super(GatherMDP, self).reset()
        self.objects = []
        existing = set()
        while len(self.objects) < self.n_apples:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = APPLE
            self.objects.append((x, y, typ))
            existing.add((x, y))
        while len(self.objects) < self.n_apples + self.n_bombs:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = BOMB
            self.objects.append((x, y, typ))
            existing.add((x, y))

        self.inner_env.reset()
        return self.get_current_obs()

    def step(self, action):
        _, _, done, info = self.inner_env.step(action)
        if done:
            return Step(self.get_current_obs(), -10, done, **info)
        com = self.inner_env.get_body_com("torso")
        x, y = com[:2]
        reward = 0
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward = reward + 1
                else:
                    reward = reward - 1
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0
        return Step(self.get_current_obs(), reward, done, **info)

    def get_readings(self):
        # compute sensor readings
        # first, obtain current orientation
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.inner_env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins
        ori = self.get_ori()
        # print ori*180/math.pi
        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb
                ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            #    ((angle + half_span) +
            #     ori) % (2 * math.pi) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity
        return apple_readings, bomb_readings

    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.inner_env.get_current_obs()
        apple_readings, bomb_readings = self.get_readings()
        return np.concatenate([self_obs, apple_readings, bomb_readings])

    def get_viewer(self):
        if self.inner_env.viewer is None:
            self.inner_env.viewer = GatherViewer(self)
            self.inner_env.viewer.start()
            self.inner_env.viewer.set_model(self.inner_env.model)
        return self.inner_env.viewer

    @property
    @overrides
    def action_space(self):
        return self.inner_env.action_space

    @property
    def action_bounds(self):
        return self.inner_env.action_bounds

    @property
    def viewer(self):
        return self.inner_env.viewer

    @property
    @overrides
    def observation_space(self):
        dim = self.inner_env.observation_space.flat_dim
        newdim = dim + self.n_bins * 2
        ub = BIG * np.ones(newdim)
        return spaces.Box(ub * -1, ub)

    def action_from_key(self, key):
        return self.inner_env.action_from_key(key)

    def render(self):
        self.get_viewer()
        self.inner_env.render()

    def get_ori(self): # get orientation
        return self.inner_env.model.data.qpos[self.__class__.ORI_IND]
