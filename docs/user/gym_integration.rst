.. _gym_integration:



===========================
Integrating with OpenAI Gym
===========================

`OpenAI Gym <https://gym.openai.com/>`_ is a recently released reinforcement learning toolkit that contains a wide
range of environments and an online scoreboard. rllab now provides a wrapper to run algorithms in rllab on environments
from OpenAI Gym, as well as submitting the results to the scoreboard. The example script in :code:`examples/trpo_gym_pendulum.py`
provides a simple example of training an agent on the :code:`Pendulum-v0` environment. The content of the file is as follows:


.. code-block:: python

    from rllab.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from rllab.envs.gym_env import GymEnv
    from rllab.envs.normalized_env import normalize
    from rllab.misc.instrument import run_experiment_lite
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


    def run_task(*_):
        env = normalize(GymEnv("Pendulum-v0"))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(8, 8)
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=env.horizon,
            n_itr=50,
            discount=0.99,
            step_size=0.01,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
        )
        algo.train()


    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # plot=True,
    )


Running the script will automatically record the episodic total reward and
periodically record video. When the script finishes running, you will see an
instruction of how to upload it to the online scoreboard, similar to the following
text (you will need to first register for an account on https://gym.openai.com,
and set the environment variable :code:`OPENAI_GYM_API_KEY` to be your API key):


.. code-block:: bash

    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python scripts/submit_gym.py data/local/experiment/experiment_2016_04_27_18_32_31_0001/gym_log

    ***************************


Comparison between rllab and OpenAI Gym
=======================================

Both rllab and OpenAI Gym set out to be frameworks for developing and evaluating reinforcement learning algorithms.

OpenAI Gym has a wider range of supported environments, as well as an online scoreboard for sharing the training results.
It makes no assumptions of how the agent should be implemented.

rllab offers a set of built-in implementations of RL algorithms. These implementations are agnostic how the environment
or the policy is laid out, as well as fine grained components for developing and experimenting with new reinforcement
learning algorithms. rllab is fully compatible with OpenAI Gym. The rllab reference implementations of a wide range of
RL algorithms enable faster experimentation and rllab provides seamless upload to Gym’s scoreboard.
