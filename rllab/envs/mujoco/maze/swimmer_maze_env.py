from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv


class SwimmerMazeEnv(MazeEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 4
    MAZE_MAKE_CONTACTS = True

