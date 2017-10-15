from rllab.algos.dqn_3 import DQN
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
import gym

env_name = 'CartPole-v1'
env = gym.make(env_name)

algo = DQN(
    env=env,
    batch_size=32,
    episodes=1000,
    gamma = 0.95,    # discount rate
    epsilon = 1.0,  # exploration rate
    epsilon_min = 0.01,
    epsilon_decay = 0.995,
    learning_rate = 0.001,
)
algo.train()
