.. _implement_algo_basic:

===================================
Implementing New Algorithms (Basic)
===================================

In this section, we will walk through the implementation of the classical
REINFORCE [1]_ algorithm, also known as the "vanilla" policy gradient.
We will start with an implementation that works with a fixed policy and
environment. The next section :ref:`implement_algo_advanced` will improve upon this
implementation, utilizing the functionalities provided by the framework to make
it more structured and command-line friendly.

Preliminaries
=============

First, let's briefly review the algorithm along with some notations. We work
with an MDP defined by the tuple :math:`(\mathcal{S}, \mathcal{A}, P, r, \mu_0, \gamma, T)`, where
:math:`\mathcal{S}` is a set of states, :math:`\mathcal{A}` is a set of
actions, :math:`P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]`
is the transition probability, :math:`r: \mathcal{S} \times \mathcal{A}
\to \mathbb{R}` is the reward function, :math:`\mu_0: \mathcal{S} \to [0, 1]`
is the initial state distribution, :math:`\gamma \in [0, 1]` is the discount
factor, and :math:`T \in \mathbb{N}` is the horizon. REINFORCE directly
optimizes a parameterized stochastic policy
:math:`\pi_\theta: \mathcal{S} \times \mathcal{A} \to [0, 1]` by performing
gradient ascent on the expected return objective:

.. math::
    
    \eta(\theta) = \mathbb{E}\left[\sum_{t=0}^T \gamma^t r(s_t, a_t)\right]

where the expectation is implicitly taken over all possible trajectories,
following the sampling procedure :math:`s_0 \sim \mu_0`,
:math:`a_t \sim \pi_\theta(\cdot | s_t)`, and
:math:`s_{t+1} \sim P(\cdot | s_t, a_t)`. By the likelihood ratio trick,
the gradient of the objective with respect to :math:`\theta` is given by

.. math::
    
    \nabla_\theta \eta(\theta) = \mathbb{E}\left[\left(\sum_{t=0}^T \gamma^t r(s_t, a_t)\right) \left(\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \right)\right]

We can reduce the variance of this estimator by noting that for :math:`t' < t`,

.. math::

    \mathbb{E}\left[ r(s_{t'}, a_{t'}) \nabla_\theta \log \pi_\theta(a_t | s_t) \right] = 0

Hence,

.. math::
    
    \nabla_\theta \eta(\theta) = \mathbb{E}\left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \sum_{t'=t}^T \gamma^{t'} r(s_{t'}, a_{t'}) \right]

Often, we use the following estimator instead:

.. math::
    
    \nabla_\theta \eta(\theta) = \mathbb{E}\left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \sum_{t'=t}^T \gamma^{t'-t} r(s_{t'}, a_{t'}) \right]

where :math:`\gamma^{t'}` is replaced by :math:`\gamma^{t'-t}`. When viewing the discount factor as a variance reduction factor for the undiscounted objective, this alternative gradient estimator has less bias, at the expense of having a larger variance. We define :math:`R_t := \sum_{t'=t}^T \gamma^{t'-t} r(s_{t'}, a_{t'})` as the empirical discounted return.

.. We can further reduce the variance by subtracting a baseline :math:`b(s_t)` from the empirical return :math:`\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'}, a_{t'})`:

.. .. math::
    
    \nabla_\theta \eta(\theta) = \mathbb{E}\left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \left(\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'}, a_{t'}) - b(s_{t}) \right) \right]

.. The baseline :math:`b(s_t)` is typically implemented as an estimator of :math:`V^\pi(s_t)`.



The above formula will be the central object of our implementation. The pseudocode for the whole algorithm is as below:

- Initialize policy :math:`\pi` with parameter :math:`\theta_1`.

- For iteration :math:`k = 1, 2, \ldots`:

    - Sample N trajectories :math:`\tau_1`, ..., :math:`\tau_n` under the
      current policy :math:`\theta_k`, where
      :math:`\tau_i = (s_t^i, a_t^i, R_t^i)_{t=0}^{T-1}`. Note that the last
      state is dropped since no action is taken after observing the last state.

    - Compute the empirical policy gradient:

    .. math::
        \widehat{\nabla_\theta \eta(\theta)} = \frac{1}{NT} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) R_t^i 

    - Take a gradient step: :math:`\theta_{k+1} = \theta_k + \alpha \widehat{\nabla_\theta \eta(\theta)}`.

Setup
=====

As a start, we will try to solve the cartpole balancing task using a neural
network policy. We will later generalize our algorithm to accept configuration
parameters. But let's keep things simple for now.

.. code-block:: python

    from __future__ import print_function
    from rllab.envs.box2d.cartpole_env import CartpoleEnv
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from rllab.envs.normalized_env import normalize
    import numpy as np
    import theano
    import theano.tensor as TT
    from lasagne.updates import adam

    # normalize() makes sure that the actions for the environment lies
    # within the range [-1, 1] (only works for environments with continuous actions)
    env = normalize(CartpoleEnv())
    # Initialize a neural network policy with a single hidden layer of 8 hidden units
    policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,))

    # We will collect 100 trajectories per iteration
    N = 100
    # Each trajectory will have at most 100 time steps
    T = 100
    # Number of iterations
    n_itr = 100
    # Set the discount factor for the problem
    discount = 0.99
    # Learning rate for the gradient update
    learning_rate = 0.01


Collecting Samples
==================

Now, let's collect samples for the environment under our current policy within a single
iteration.

.. code-block:: python

    paths = []

    for _ in xrange(N):
        observations = []
        actions = []
        rewards = []

        observation = env.reset()

        for _ in xrange(T):
            # policy.get_action() returns a pair of values. The second one returns a dictionary, whose values contains
            # sufficient statistics for the action distribution. It should at least contain entries that would be
            # returned by calling policy.dist_info(), which is the non-symbolic analog of policy.dist_info_sym().
            # Storing these statistics is useful, e.g., when forming importance sampling ratios. In our case it is
            # not needed.
            action, _ = policy.get_action(observation)
            # Recall that the last entry of the tuple stores diagnostic information about the environment. In our
            # case it is not needed.
            next_observation, reward, terminal, _ = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            observation = next_observation
            if terminal:
                # Finish rollout if terminal state reached
                break

        # We need to compute the empirical return for each time step along the
        # trajectory
        returns = []
        return_so_far = 0
        for t in xrange(len(rewards) - 1, -1, -1):
            return_so_far = rewards[t] + discount * return_so_far
            returns.append(return_so_far)
        # The returns are stored backwards in time, so we need to revert it
        returns = returns[::-1]

        paths.append(dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            returns=np.array(returns)
        ))

Observe that according to the formula for the empirical policy gradient, we
could concatenate all the collected data for different trajectories together,
which helps us vectorize the implementation further.

.. code-block:: python

    observations = np.concatenate([p["observations"] for p in paths])
    actions = np.concatenate([p["actions"] for p in paths])
    returns = np.concatenate([p["returns"] for p in paths])

Constructing the Computation Graph
==================================

We will use `Theano <http://deeplearning.net/software/theano/>`_ for our
implementation, and we assume that the reader has some familiarity with it.
If not, it would be good to go through `some tutorials <http://nbviewer.jupyter.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb>`_
first.

First, we construct symbolic variables for the input data:

.. code-block:: python

    # Create a Theano variable for storing the observations
    # We could have simply written `observations_var = TT.matrix('observations')` instead for this example. However,
    # doing it in a slightly more abstract way allows us to delegate to the environment for handling the correct data
    # type for the variable. For instance, for an environment with discrete observations, we might want to use integer
    # types if the observations are represented as one-hot vectors.
    observations_var = env.observation_space.new_tensor_variable(
        'observations',
        # It should have 1 extra dimension since we want to represent a list of observations
        extra_dims=1
    )
    actions_var = env.action_space.new_tensor_variable(
        'actions',
        extra_dims=1
    )
    returns_var = TT.vector('returns')

Note that we can transform the policy gradient formula as

.. math::

    \widehat{\nabla_\theta \eta(\theta)} = \nabla_\theta \left( \frac{1}{NT} \sum_{i=1}^N \sum_{t=0}^{T-1} \log \pi_\theta(a_t^i | s_t^i) R_t^i \right) = \nabla_\theta L(\theta)

where :math:`L(\theta) = \frac{1}{NT} \sum_{i=1}^N \sum_{t=0}^{T-1} \log \pi_\theta(a_t^i | s_t^i) R_t^i` is called the surrogate function. Hence, we can first construct the computation graph for :math:`L(\theta)`, and then take its gradient to get the empirical policy gradient.

.. code-block:: python

    # policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
    # distribution of the actions. For a Gaussian policy, it contains the mean and (log) standard deviation.
    dist_info_vars = policy.dist_info_sym(observations_var, actions_var)

    # policy.distribution returns a distribution object under rllab.distributions. It contains many utilities for computing
    # distribution-related quantities, given the computed dist_info_vars. Below we use dist.log_likelihood_sym to compute
    # the symbolic log-likelihood. For this example, the corresponding distribution is an instance of the class
    # rllab.distributions.DiagonalGaussian
    dist = policy.distribution

    # Note that we negate the objective, since most optimizers assume a
    # minimization problem
    surr = - TT.mean(dist.log_likelihood_sym(actions_var, dist_info_vars) * returns_var)

    # Get the list of trainable parameters.
    params = policy.get_params(trainable=True)
    grads = theano.grad(surr, params)

Gradient Update and Diagnostics
===============================

We are almost done! Now, you can use your favorite stochastic optimization algorithm for performing the parameter update. We choose ADAM [2]_ in our implementation:

.. code-block:: python

    f_train = theano.function(
        inputs=[observations_var, actions_var, returns_var],
        outputs=None,
        updates=adam(grads, params, learning_rate=learning_rate),
        allow_input_downcast=True
    )
    f_train(observations, actions, returns)

Since this algorithm is on-policy, we can evaluate its performance by inspecting the collected samples:

.. code-block:: py

    print('Average Return:', np.mean([sum(path["rewards"]) for path in paths]))

The complete source code so far is available at :code:`examples/vpg_1.py`.

Additional Tricks
=================

Adding a Baseline
-----------------

The variance of the policy gradient can be further reduced by adding a baseline. The refined formula is given by

.. math::
    \widehat{\nabla_\theta \eta(\theta)} = \frac{1}{NT} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) (R_t^i - b(s_t^i))

We can do this since :math:`\mathbb{E} \left[\nabla_\theta \log \pi_\theta(a_t^i | s_t^i) b(s_t^i)\right] = 0`

The baseline is typically implemented as an estimator of :math:`V^\pi(s)`. In
this case, :math:`R_t^i - b(s_t^i)` is an estimator of
:math:`A^\pi(s_t^i, a_t^i)`. The framework implements a few options for the
baseline. A good balance of computational efficiency and accuracy is achieved
by a linear baseline using state features, available
at :code:`rllab/baselines/linear_feature_baseline.py`. To use it in our implementation,
the relevant code looks like the following:

.. code-block:: python

    # ... initialization code ...

    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    baseline = LinearFeatureBaseline(env.spec)

    # ... inside the loop for each episode, after the samples are collected

    path = dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
    )

    path_baseline = baseline.predict(path)
    advantages = []
    returns = []
    return_so_far = 0
    for t in xrange(len(rewards) - 1, -1, -1):
        return_so_far = rewards[t] + discount * return_so_far
        returns.append(return_so_far)
        advantage = return_so_far - path_baseline[t]
        advantages.append(advantage)
    # The advantages are stored backwards in time, so we need to revert it
    advantages = np.array(advantages[::-1])
    # And we need to do the same thing for the list of returns
    returns = np.array(returns[::-1])

Normalizing the returns
-----------------------

Currently, the learning rate we set for the algorithm is very susceptible to
reward scaling. We can alleviate this dependency by whitening the advantages
before computing the gradients. In terms of code, this would be:

.. code-block:: py

    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

Training the baseline
---------------------

After each iteration, we use the newly collected trajectories to train our baseline:

.. code-block:: py

    baseline.fit(paths)

The reason that this is executed after computing the baselines along the given
trajectories is that in the extreme case, if we only have one trajectory starting
from each state, and if the baseline could fit the data perfectly, then all the
advantages would be zero, giving us no gradient signals at all.

Now, we can train the policy much faster (we need to change the learning rate
accordingly because of the rescaling). The complete source code so far is
available at :code:`examples/vpg_2.py`

.. [1] Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
.. [2] Kingma, Diederik P., and Jimmy Ba Adam. "A method for stochastic optimization." International Conference on Learning Representation. 2015.
