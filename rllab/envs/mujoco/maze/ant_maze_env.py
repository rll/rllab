from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import math


class AntMazeEnv(MazeEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 3

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0

    def get_ori(self):
        ori = [0, 1, 0, 0] 
        rot = self.wrapped_env.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND+4] # take the quaternion
        ori = q_mult(q_mult(rot,ori),q_inv(rot))[1:3] # project onto x-y plane
        ori = math.atan2(ori[1],ori[0]) 
        return ori
