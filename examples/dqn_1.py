from rllab.algos.dqn import DQN
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
import gym

env_name = "MsPacman-v0"
env = gym.make(env_name)

algo = DQN(
    env=env,
    num_episodes=400,
    save_name=env_name,
    # epsilon=0.1,
    # batch_size=32,
    # discount=0.99,
    # memory=50,
)
algo.train()
