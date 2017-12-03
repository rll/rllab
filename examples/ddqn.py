from rllab.algos.ddqn import DDQN
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
import gym

env_name = 'CartPole-v1'
env = gym.make(env_name)

algo = DDQN(
    env=env,
    batch_size=32,
    episodes=5000,
    gamma = 0.95,    # discount rate
    epsilon = 1.0,  # exploration rate
    epsilon_min = 0.01,
    epsilon_decay = 0.99,
    learning_rate = 0.001,
)
algo.train()
