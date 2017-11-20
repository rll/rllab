.. _experiments:


===================
Running Experiments
===================


We use object oriented abstractions for different components required for an experiment. To run an experiment, simply construct the corresponding objects for the environment, algorithm, etc. and call the appropriate train method on the algorithm. A sample script is provided in :code:`examples/trpo_cartpole.py`. The code is also pasted below for a quick glance:

.. code-block:: python

    from rllab.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from rllab.envs.box2d.cartpole_env import CartpoleEnv
    from rllab.envs.normalized_env import normalize
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

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
        whole_paths=True,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        step_size=0.01,
    )
    algo.train()


Running the script for the first time might take a while for initializing
Theano and compiling the computation graph, which can take a few minutes.
Subsequent runs will be much faster since the compilation is cached. You should
see some log messages like the following:

.. code-block:: text

    using seed 1
    instantiating rllab.envs.box2d.cartpole_env.CartpoleEnv
    instantiating rllab.policy.mean_std_nn_policy.MeanStdNNPolicy
    using argument hidden_sizes with value [32, 32]
    instantiating rllab.baseline.linear_feature_baseline.LinearFeatureBaseline
    instantiating rllab.algo.trpo.TRPO
    using argument batch_size with value 4000
    using argument whole_paths with value True
    using argument n_itr with value 40
    using argument step_size with value 0.01
    using argument discount with value 0.99
    using argument max_path_length with value 100
    using seed 0
    0%                          100%
    [##############################] | ETA: 00:00:00
    Total time elapsed: 00:00:02
    2016-02-14 14:30:56.631891 PST | [trpo_cartpole] itr #0 | fitting baseline...
    2016-02-14 14:30:56.677086 PST | [trpo_cartpole] itr #0 | fitted
    2016-02-14 14:30:56.682712 PST | [trpo_cartpole] itr #0 | optimizing policy
    2016-02-14 14:30:56.686587 PST | [trpo_cartpole] itr #0 | computing loss before
    2016-02-14 14:30:56.698566 PST | [trpo_cartpole] itr #0 | performing update
    2016-02-14 14:30:56.698676 PST | [trpo_cartpole] itr #0 | computing descent direction
    2016-02-14 14:31:26.241657 PST | [trpo_cartpole] itr #0 | descent direction computed
    2016-02-14 14:31:26.241828 PST | [trpo_cartpole] itr #0 | performing backtracking
    2016-02-14 14:31:29.906126 PST | [trpo_cartpole] itr #0 | backtracking finished
    2016-02-14 14:31:29.906335 PST | [trpo_cartpole] itr #0 | computing loss after
    2016-02-14 14:31:29.912287 PST | [trpo_cartpole] itr #0 | optimization finished
    2016-02-14 14:31:29.912483 PST | [trpo_cartpole] itr #0 | saving snapshot...
    2016-02-14 14:31:29.914311 PST | [trpo_cartpole] itr #0 | saved
    2016-02-14 14:31:29.915302 PST | -----------------------  -------------
    2016-02-14 14:31:29.915365 PST | Iteration                   0
    2016-02-14 14:31:29.915410 PST | Entropy                     1.41894
    2016-02-14 14:31:29.915452 PST | Perplexity                  4.13273
    2016-02-14 14:31:29.915492 PST | AverageReturn              68.3242
    2016-02-14 14:31:29.915533 PST | StdReturn                  42.6061
    2016-02-14 14:31:29.915573 PST | MaxReturn                 369.864
    2016-02-14 14:31:29.915612 PST | MinReturn                  19.9874
    2016-02-14 14:31:29.915651 PST | AverageDiscountedReturn    65.5314
    2016-02-14 14:31:29.915691 PST | NumTrajs                 1278
    2016-02-14 14:31:29.915730 PST | ExplainedVariance           0
    2016-02-14 14:31:29.915768 PST | AveragePolicyStd            1
    2016-02-14 14:31:29.921911 PST | BacktrackItr                2
    2016-02-14 14:31:29.922008 PST | MeanKL                      0.00305741
    2016-02-14 14:31:29.922054 PST | MaxKL                       0.0360272
    2016-02-14 14:31:29.922096 PST | LossBefore                 -0.0292939
    2016-02-14 14:31:29.922146 PST | LossAfter                  -0.0510883
    2016-02-14 14:31:29.922186 PST | -----------------------  -------------


Pickled Mode Experiments
=====================

:code:`rllab` also supports a "pickled" mode for running experiments, which supports more configurations like logging and parallelization. A sample script is provided in :code:`examples/trpo_cartpole_pickled.py`. The content is pasted below:

.. code-block:: python

    from rllab.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from rllab.envs.box2d.cartpole_env import CartpoleEnv
    from rllab.envs.normalized_env import normalize
    from rllab.misc.instrument import run_experiment_lite
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


    def run_task(*_):
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
            n_itr=1000,
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


Note that the execution of the experiment (including the construction of relevant objects, like environment, policy, algorithm, etc.) has been wrapped in a function call, which is then passed to the `run_experiment_lite` method, which serializes the fucntion call, and launches a script that actually runs the experiment.

The benefit for launching experiment this way is that we separate the configuration of experiment parameters and the actual execution of the experiment. `run_experiment_lite` supports multiple ways of running the experiment, either locally, locally in a docker container, or remotely on ec2 (see the section on :ref:`cluster`). Multiple experiments with different hyper-parameter settings can be quickly constructed and launched simultaneously on multiple ec2 machines using this abstraction.

Another subtle point is that we use Theano for our algorithm implementations, which has rather poor support for mixed GPU and CPU usage. This might be handy when the main process wants to use GPU for the batch optimization phase, while multiple worker processes want to use the CPU for generating trajectory rollouts. Launching the experiment separately allows the worker processes to be properly initialized with Theano configured to use CPU.

Additional arguments for `run_experiment_lite` (experimental):

- `exp_name`: If this is set, the experiment data will be stored in the folder `data/local/{exp_name}`. By default, the folder name is set to `experiment_{timestamp}`.
- `exp_prefix`: If this is set, and if `exp_name` is not specified, the experiment folder name will be set to `{exp_prefix}_{timestamp}`.
