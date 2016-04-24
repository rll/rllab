.. _implement_mdp:

=============================
Implementing New Environments
=============================

In this section, we will walk through an example of implementing a point robot
environment using our framework.

Each environment should implement at least the following methods / properties defined
in the file :code:`rllab/envs/base.py`:

.. code-block:: python

    class Env(object):
        def step(self, action):
            """
            Run one timestep of the environment's dynamics. When end of episode
            is reached, reset() should be called to reset the environment's internal state.
            Input
            -----
            action : an action provided by the environment
            Outputs
            -------
            (observation, reward, done, info)
            observation : agent's observation of the current environment
            reward [Float] : amount of reward due to the previous action
            done : a boolean, indicating whether the episode has ended
            info : a dictionary containing other diagnostic information from the previous action
            """
            raise NotImplementedError

        def reset(self):
            """
            Resets the state of the environment, returning an initial observation.
            Outputs
            -------
            observation : the initial observation of the space. (Initial reward is assumed to be 0.)
            """
            raise NotImplementedError

        @property
        def action_space(self):
            """
            Returns a Space object
            """
            raise NotImplementedError

        @property
        def observation_space(self):
            """
            Returns a Space object
            """
            raise NotImplementedError


We will implement a simple environment with 2D observations and 2D actions. The goal is
to control a point robot in 2D to move it to the origin. We receive position of
a point robot in the 2D plane :math:`(x, y) \in \mathbb{R}^2`. The action is
its velocity :math:`(\dot x, \dot y) \in \mathbb{R}^2` constrained so that
:math:`|\dot x| \leq 0.1` and :math:`|\dot y| \leq 0.1`. We encourage the robot
to move to the origin by defining its reward as the negative distance to the
origin: :math:`r(x, y) = - \sqrt{x^2 + y^2}`.

We start by creating a new file for the environment. We assume that it is placed under
:code:`examples/point_env.py`. First, let's declare a class inheriting from
the base environment and add some imports:

.. code-block:: python

    from rllab.envs.base import Env
    from rllab.envs.base import Step
    from rllab.spaces import Box
    import numpy as np


    class PointEnv(Env):

        # ...

For each environment, we will need to specify the set of valid observations and the
set of valid actions. This is done by implementing the following
property methods:

.. code-block:: python

    class PointEnv(Env):

        # ...

        @property
        def observation_space(self):
            return Box(low=-np.inf, high=np.inf, shape=(2,))

        @property
        def action_space(self):
            return Box(low=-0.1, high=0.1, shape=(2,))

The :code:`Box` space means that the observations and actions are 2D vectors
with continuous values. The observations can have arbitrary values, while the
actions should have magnitude at most 0.1.

Now onto the interesting part, where we actually implement the dynamics for the
MDP. This is done through two methods, :code:`reset` and
:code:`step`. The :code:`reset` method randomly initializes the state
of the environment according to some initial state distribution. To keep things
simple, we will just sample the coordinates from a uniform distribution. The
method should also return the initial observation. In our case, it will be the
same as its state.

.. code-block:: python

    class PointEnv(Env):

        # ...

        def reset(self):
            self._state = np.random.uniform(-1, 1, size=(2,))
            observation = np.copy(self._state)
            return observation

The :code:`step` method takes an action and advances the state of the
environment. It should return a :code:`Step` object (which is a wrapper around
:code:`namedtuple`), containing the observation for the next time step, the reward,
a flag indicating whether the episode is terminated after taking the step, and optional
extra keyword arguments (whose values should be vectors only) for diagnostic purposes.
The procedure that interfaces with the environment is responsible for calling
:code:`reset` after seeing that the episode is terminated.

.. code-block:: python

    class PointEnv(Env):

        # ...

        def step(self, action):
            self._state = self._state + action
            x, y = self._state
            reward = - (x**2 + y**2) ** 0.5
            done = abs(x) < 0.01 and abs(y) < 0.01
            next_observation = np.copy(self._state)
            return Step(observation=next_observation, reward=reward, done=done)

Finally, we can implement some plotting to visualize what the MDP is doing. For
simplicity, let's just print the current state of the MDP on the terminal:

.. code-block:: python

    class PointEnv(Env):

        # ...

        def render(self):
            print 'current state:', self._state

And we're done! We can now simulate the environment using the following diagnostic
script:

.. code-block:: bash

    python scripts/sim_env.py examples.point_env --mode random

It simulates an episode of the environment with random actions, sampled from a
uniform distribution within the defined action bounds.

You could also train a neural network policy to solve the task, which is probably
an overkill. To do so, create a new script with the following content (we will use
stub mode):


.. code-block:: python

    from rllab.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from examples.point_env import PointEnv
    from rllab.envs.normalized_env import normalize
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

    env = normalize(PointEnv())
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
    )
    algo.train()

Assume that the file is :code:`examples/trpo_point.py`. You can then run the script:

.. code-block:: bash

    python examples/trpo_point.py
