from rllab.envs.mujoco.gather.gather_env import GatherEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv


class SwimmerGatherEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2
