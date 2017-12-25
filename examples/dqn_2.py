from rllab.algos.dqn_2 import DQN
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
import gym

env_name = 'Breakout-v0'
env = gym.make(env_name)

algo = DQN(
    env=env,
    env_name=env_name,
    frame_width=84,  # Resized frame width
    frame_height=84,  # Resized frame height
    num_episodes = 12000,  # Number of episodes the agent plays
    state_length = 4,  # Number of most recent frames to produce the input to the network
    gamma = 0.99,  # Discount factor
    exploration_steps = 1000000,  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
    initial_epsilon = 1.0,  # Initial value of epsilon in epsilon-greedy
    final_epsilon = 0.1,  # Final value of epsilon in epsilon-greedy
    initial_replay_size = 20000,  # Number of steps to populate the replay memory before training starts
    num_replay_memory = 400000,  # Number of replay memory the agent uses for training
    batch_size = 32,  # Mini batch size
    target_update_interval = 10000,  # The frequency with which the target network is updated
    train_interval = 4,  # The agent selects 4 actions between successive updates
    learning_rate = 0.00025,  # Learning rate used by RMSProp
    momentum = 0.95,  # Momentum used by RMSProp
    min_grad = 0.01,  # Constant added to the squared gradient in the denominator of the RMSProp update
    save_interval = 300000,  # The frequency with which the network is saved
    no_op_steps = 30,  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
    load_network = False,
    save_network_path = './saved_networks/' + env_name,
    save_summary_path = './summary/' + env_name,
)
algo.train()
