from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy


map_desc = '4x4' # map description, see multi_agent_grid_world_env.py
n_row = 4  # n_row and n_col need to be compatible with desc
n_col = 4

env = TfEnv(normalize(GridWorldEnv(desc = map_desc)))

policy = CategoricalMLPPolicy(
    'MAP',
    env_spec=env.spec,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=3000,
    max_path_length=20,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
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
