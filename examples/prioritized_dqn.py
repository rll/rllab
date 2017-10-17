from rllab.algos.prioritized_dqn import PrioritizedReplayDQN
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
import gym

env_name = 'Pendulum-v0'
env = gym.make(env_name)

algo = PrioritizedReplayDQN(
    env=env,
    memory_size=10000,
    learning_rate=0.005,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=500,
    memory_size_agent=10000,
    batch_size=32,
)
algo.train()
