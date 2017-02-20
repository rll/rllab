from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.multi_agent_grid_world_env import MultiAgentGridWorldEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.multi_agent_categorical_policy import MultiAgentCategoricalMLPPolicy


map_desc = '4x4-fix' # map description, see multi_agent_grid_world_env.py
n_row = 4  # n_row and n_col need to be compatible with desc
n_col = 4
n_agent = 1 # 2 <= agents <= 6

env = TfEnv(normalize(MultiAgentGridWorldEnv(n = n_agent, desc = map_desc)))

policy = MultiAgentCategoricalMLPPolicy(
    'MAP',
    n_row,
    n_col,
    n_agent,
    env_spec=env.spec,
    feature_dim = 10, # feature from each agent's local information
    msg_dim = 0, # when msg_dim == 0, no communication
    conv_layers=[4,4,4], # number of conv-layers and the number of kernels
    hidden_layers=[], # hidden layers after conv-layers
    comm_layers = [], # hidden layers after receiving msgs from other agents
    act_dim = 5 # always 5 in grid domain: 4 directions + stay
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=500,
    max_path_length=20,
    n_itr=80,
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
