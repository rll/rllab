from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.multi_agent_grid_world_env import MultiAgentGridWorldEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.multi_agent_categorical_policy import MultiAgentCategoricalMLPPolicy

# store policy
import rllab.misc.logger as logger

policy_dir = '../temp/' # or None
store_mode = 'gap' # 'all' or 'last' or None
store_gap = 5

map_desc = 'chain-tiny-fix' # map description, see multi_agent_grid_world_env.py
n_row = 3  # n_row and n_col need to be compatible with desc
n_col = 5
n_agent = 2 # 2 <= agents <= 6


logger.set_snapshot_dir(policy_dir)
logger.set_snapshot_mode(store_mode)
logger.set_snapshot_gap(store_gap)

env = TfEnv(normalize(MultiAgentGridWorldEnv(n = n_agent, desc = map_desc)))

policy = MultiAgentCategoricalMLPPolicy(
    'MAP',
    n_row,
    n_col,
    n_agent,
    env_spec=env.spec,
    feature_dim = 10, # feature from each agent's local information
    msg_dim = 1, # when msg_dim == 0, no communication
    conv_layers= [], # number of conv-layers and the number of kernels
    hidden_layers=[10,10], # hidden layers after conv-layers
    comm_layers = [10], # hidden layers after receiving msgs from other agents
    act_dim = 5, # always 5 in grid domain: 4 directions + stay
    shared_weights = False,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=2000,
    max_path_length=30,
    n_itr=100,
    discount=0.99,
    step_size=0.1,
)
"""
algo = VPG(
    env = env,
    policy = policy,
    baseline = baseline,
    batch_size = 3000,
    max_path_length=20,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)
"""
algo.train()
