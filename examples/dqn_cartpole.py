from rllab.algos.dqn import DQN
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize

env = normalize(CartpoleEnv())

algo = DQN(
    env=env,
    num_episodes=400,
    state_size=(5,5,5),
    number_of_actions=3,
    # epsilon=0.1,
    # batch_size=32,
    # discount=0.99,
    # memory=50,
)
algo.train()
