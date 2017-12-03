from rllab.algos.dueling_dqn import DuelingDQN
import gym

env_name = 'MountainCar-v0'
env = gym.make(env_name)

algo = DuelingDQN(
    env=env,
    memory_size=10000,
    action_space=3,
    learning_rate=0.001,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=500,
    memory_size_2=10000,
    batch_size=32,
)
algo.train()
