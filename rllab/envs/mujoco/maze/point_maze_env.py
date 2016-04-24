from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.point_env import PointEnv


class PointMazeEnv(MazeEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0

    MANUAL_COLLISION = True
