from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import sys

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

class VG(VariantGenerator):

    @variant
    def step_size(self):
        return [0.01, 0.05, 0.1]

    @variant
    def seed(self):
        return [1, 11, 21, 31, 41]

variants = VG().variants()

for v in variants:

    env = normalize(CartpoleEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        step_size=v["step_size"],
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="first_exp",
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        # mode="local",
        mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
    sys.exit()
