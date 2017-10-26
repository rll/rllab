from rllab.algos.dqn import DQN
import gym

env_name = 'MountainCar-v0'
env = gym.make(env_name)

algo = DQN(
    env=env,
    memory_size=3000,
    action_space=3,
    learning_rate=0.001,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=200,
    memory_size_2=500,
    batch_size=32,
)
algo.train()
