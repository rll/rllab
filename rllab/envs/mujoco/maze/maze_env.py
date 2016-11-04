import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
import math

import numpy as np

from rllab import spaces
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv

from rllab.core.serializable import Serializable
from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.maze.maze_env_utils import ray_segment_intersect, point_distance



class MazeEnv(ProxyEnv, Serializable):
    MODEL_CLASS = None
    ORI_IND = None

    MAZE_HEIGHT = None
    MAZE_SIZE_SCALING = None
    MAZE_MAKE_CONTACTS = False
    MAZE_STRUCTURE = [
        [1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 'g', 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    MANUAL_COLLISION = False

    def __init__(
            self,
            n_bins=20,
            sensor_range=10.,
            sensor_span=math.pi,
            *args,
            **kwargs):

        self._n_bins = n_bins
        self._sensor_range = sensor_range
        self._sensor_span = sensor_span

        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        xml_path = osp.join(MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        size_scaling = self.__class__.MAZE_SIZE_SCALING
        height = self.__class__.MAZE_HEIGHT
        structure = self.__class__.MAZE_STRUCTURE

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) == '1':
                    # offset all coordinates so that robot starts at the origin
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1"
                    )

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        if self.__class__.MAZE_MAKE_CONTACTS:
            contact = ET.SubElement(
                tree.find("."), "contact"
            )
            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if str(structure[i][j]) == '1':
                        for geom in geoms:
                            ET.SubElement(
                                contact, "pair",
                                geom1=geom.attrib["name"],
                                geom2="block_%d_%d" % (i, j)
                            )

        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path)

        self._goal_range = self._find_goal_range()
        self._cached_segments = None

        inner_env = model_cls(*args, file_path=file_path, **kwargs)
        ProxyEnv.__init__(self, inner_env)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        ori = self.get_ori()

        #print ori*180/math.pi

        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING

        segments = []
        # compute the distance of all segments

        # Get all line segments of the goal and the obstacles
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1 or structure[i][j] == 'g':
                    cx = j * size_scaling - self._init_torso_x
                    cy = i * size_scaling - self._init_torso_y
                    x1 = cx - 0.5 * size_scaling
                    x2 = cx + 0.5 * size_scaling
                    y1 = cy - 0.5 * size_scaling
                    y2 = cy + 0.5 * size_scaling
                    struct_segments = [
                        ((x1, y1), (x2, y1)),
                        ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)),
                        ((x1, y2), (x1, y1)),
                    ]
                    for seg in struct_segments:
                        segments.append(dict(
                            segment=seg,
                            type=structure[i][j],
                        ))

        wall_readings = np.zeros(self._n_bins)
        goal_readings = np.zeros(self._n_bins)

        for ray_idx in range(self._n_bins):
            ray_ori = ori - self._sensor_span * 0.5 + 1.0 * (2 * ray_idx + 1) / (2 * self._n_bins) * self._sensor_span
            ray_segments = []
            for seg in segments:
                p = ray_segment_intersect(ray=((robot_x, robot_y), ray_ori), segment=seg["segment"])
                if p is not None:
                    ray_segments.append(dict(
                        segment=seg["segment"],
                        type=seg["type"],
                        ray_ori=ray_ori,
                        distance=point_distance(p, (robot_x, robot_y)),
                    ))
            if len(ray_segments) > 0:
                first_seg = sorted(ray_segments, key=lambda x: x["distance"])[0]
                # print first_seg
                if first_seg["type"] == 1:
                    # Wall -> add to wall readings
                    if first_seg["distance"] <= self._sensor_range:
                        wall_readings[ray_idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range
                elif first_seg["type"] == 'g':
                    # Goal -> add to goal readings
                    if first_seg["distance"] <= self._sensor_range:
                        goal_readings[ray_idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range
                else:
                    assert False

        obs = np.concatenate([
            self.wrapped_env.get_current_obs(),
            wall_readings,
            goal_readings
        ])
        # print "wall readings:", wall_readings
        # print "goal readings:", goal_readings

        return obs

    def reset(self):
        self.wrapped_env.reset()
        return self.get_current_obs()

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    @property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    def _find_robot(self):
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    return j * size_scaling, i * size_scaling
        assert False

    def _find_goal_range(self):
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'g':
                    minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
                    maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
                    miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
                    maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
                    return minx, maxx, miny, maxy

    def _is_in_collision(self, pos):
        x, y = pos
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1:
                    minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
                    maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
                    miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
                    maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
                    if minx <= x <= maxx and miny <= y <= maxy:
                        return True
        return False

    def step(self, action):
        if self.MANUAL_COLLISION:
            old_pos = self.wrapped_env.get_xy()
            _, _, done, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            if self._is_in_collision(new_pos):
                self.wrapped_env.set_xy(old_pos)
                done = False
        else:
            _, _, done, info = self.wrapped_env.step(action)
        next_obs = self.get_current_obs()
        x, y = self.wrapped_env.get_body_com("torso")[:2]
        # ref_x = x + self._init_torso_x
        # ref_y = y + self._init_torso_y
        reward = 0
        minx, maxx, miny, maxy = self._goal_range
        # print "goal range: x [%s,%s], y [%s,%s], now [%s,%s]" % (str(minx), str(maxx), str(miny), str(maxy),
        #                                                          str(x), str(y))
        if minx <= x <= maxx and miny <= y <= maxy:
            done = True
            reward = 1
        return Step(next_obs, reward, done, **info)

    def action_from_key(self, key):
        return self.wrapped_env.action_from_key(key)

    def get_ori(self):
        return self.wrapped_env.model.data.qpos[self.__class__.ORI_IND]
