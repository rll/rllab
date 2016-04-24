.. _implement_algo_advanced:

======================================
Implementing New Algorithms (Advanced)
======================================

In this section, we will walk through the implementation of vanilla policy gradient
algorithm provided in the algorithm, available at :code:`rllab/algos/vpg.py`. It utilizes
many functionalities provided by the framework, which we describe below.


The :code:`BatchPolopt` Class
=============================

The :code:`VPG` class inherits from :code:`BatchPolopt`, which is an abstract
class inherited by algorithms with a common structure. The structure is as
follows:

- Initialize policy :math:`\pi` with parameter :math:`\theta_1`.

- Initialize the computational graph structure.

- For iteration :math:`k = 1, 2, \ldots`:

    - Sample N trajectories :math:`\tau_1`, ..., :math:`\tau_n` under the
      current policy :math:`\theta_k`, where
      :math:`\tau_i = (s_t^i, a_t^i, R_t^i)_{t=0}^{T-1}`. Note that the last
      state is dropped since no action is taken after observing the last state.

    - Update the policy based on the collected on-policy trajectories.

    - Print diagnostic information and store intermediate results.

Note the parallel between the structure above and the pseudocode for VPG. The
:code:`BatchPolopt` class takes care of collecting samples and common diagnostic
information. It also provides an abstraction of the general procedure above, so
that algorithm implementations only need to fill the missing pieces. The core
of the :code:`BatchPolopt` class is the :code:`train()` method:


.. code-block:: python

    def train(self):
        # ...
        self.init_opt()
        for itr in xrange(self.start_itr, self.n_itr):
            paths = self.obtain_samples(itr)
            samples_data = self.process_samples(itr, paths)
            self.optimize_policy(itr, samples_data)
            params = self.get_itr_snapshot(itr, samples_data)
            logger.save_itr_params(itr, params)
            # ...

The methods :code:`obtain_samples` and :code:`process_samples` are implemented
for you. The derived class needs to provide implementation for :code:`init_opt`,
which initializes the computation graph, :code:`optimize_policy`, which updates
the policy based on the collected data, and :code:`get_itr_snapshot`, which
returns a dictionary of objects to be persisted per iteration.

The :code:`BatchPolopt` class powers quite a few algorithms:

- Vanilla Policy Gradient: :code:`rllab/algos/vpg.py`

- Natural Policy Gradient: :code:`rllab/algos/npg.py`

- Reward-Weighted Regression: :code:`rllab/algos/erwr.py`

- Trust Region Policy Optimization: :code:`rllab/algos/trpo.py`

- Relative Entropy Policy Search: :code:`rllab/algos/reps.py`

To give an illustration, here's how we might implement :code:`init_opt` for VPG
(the actual code in :code:`rllab/algos/vpg.py` is longer due to the need to log
extra diagnostic information as well as supporting recurrent policies):

.. code-block:: python

    from rllab.misc.ext import extract, compile_function, new_tensor

    # ...

    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantage_var = TT.vector('advantage')
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: TT.matrix('old_%s' % k)
            for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - TT.mean(logli * advantage_var)

        input_list = [obs_var, action_var, advantage_var] + state_info_vars_list

        self.optimizer.update_opt(surr_obj, target=self.policy, inputs=input_list)


The code is very similar to what we implemented in the basic version. Note that
we use an optimizer, which in this case would be an instance of :code:`rllab.optimizers.first_order_optimizer.FirstOrderOptimizer`.

Here's how we might implement :code:`optimize_policy`:

.. code-block:: python

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        self.optimizer.optimize(inputs)


Parallel Sampling
=================

The :code:`rllab.parallel_sampler` module takes care of parallelizing the
sampling process and aggregating the collected trajectory data. It is used
by the :code:`BatchPolopt` class like below:

.. code-block:: python

    # At the beginning of training, we need to register the environment and the policy
    # onto the parallel_sampler
    parallel_sampler.populate_task(self.env, self.policy)

    # ...

    # Within each iteration, we just need to update the policy parameters to
    # each worker
    cur_params = self.policy.get_param_values()

    paths = parallel_sampler.request_samples(
        policy_params=cur_params,
        max_samples=self.batch_size,
        max_path_length=self.max_path_length,
    )

The returned :code:`paths` is a list of dictionaries with keys :code:`rewards`,
:code:`observations`, :code:`actions`, :code:`env_infos`, and :code:`agent_infos`.
The latter two, :code:`env_infos` and :code:`agent_infos` are in turn dictionaries,
whose values are numpy arrays of the environment and agent (policy) information
per time step stacked together. :code:`agent_infos` will contain at least information
that would be returned by calling :code:`policy.dist_info()`. For a gaussian
distribution with diagonal variance, this would be the means and the logarithm
of the standard deviations.

After collecting the trajectories, the :code:`process_samples` method in the
:code:`BatchPolopt` class computes the empirical returns and advantages by
using the baseline specified through command line arguments (we'll talk about
this below). Then it trains the baseline using the collected data, and
concatenates all rewards, observations, etc. together to form a single huge
tensor, just as we did for the basic algorithm implementation.

One different semantics from the basic implementation is that, rather than
collecting a fixed number of trajectories with potentially different number
of steps per trajectory (if the environment implements a termination condition), we
specify a desired total number of samples (i.e. time steps) per iteration. The
number of actual samples collected will be around this number, although sometimes
slightly larger, to make sure that all trajectories are run until either the
horizon or the termination condition is met.
