# 2016-05-21

- Updated Distribution class to include dimensions.

# 2016-04-24

- Updated Mujoco to v1.31

# 2016-04-21

- Fixed `tensor_utils.concat_tensor_dict_list` to handle nested situations properly.

# 2016-04-20

- Default nonlinearity for `CategoricalMLPPolicy` changed to `tanh` as well, for consistency.
- Add `flatten_n`, `unflatten_n` support for `Discrete` and `Product` spaces.
- Changed `dist_info_sym` and `dist_info` interface for policies. Previously it takes both the observations and actions as input arguments, where actions are needed for recurrent policies when the policy takes both the current state and the previous action into account. However this is rather artificial. The interface is now changed to take in the observation plus a dictionary of state-related information. An extra property `state_info_keys` is added to specify the list of keys used for state-related information. By default this is an empty list.
- Removed `lasagne_recurrent.py` since it's not used anywhere, and its functionality is replaced by `GRUNetwork` implemented in `rllab.core.network`.

# 2016-04-17

- Restored the default value of the `whole_paths` parameter in `BatchPolopt` back to `True`. This is more consistent with previous configurations.

# 2016-04-16

- Removed the helper method `rllab.misc.ext.merge_dict`. Turns out Python's `dict` constructor already supports this functionality: `merge_dict(dict1, dict2) == dict(dict1, **dict2)`.
- Added a `min_std` option to `GaussianMLPPolicy`. This avoids the gradients being unstable near deterministic policies.

# 2016-04-11

- Added a method `truncate_paths` to the `rllab.sampler.parallel_sampler` module. This should be sufficient to replace the old configurable parameter `whole_paths` which has been removed during refactoring.

# 2016-04-10

- Known issues:
  - TRPO does not work well with relu since the hessian is undefined at 0, causing NaN sometimes. This issue of Theano is tracked here: https://github.com/Theano/Theano/issues/4353). If relu must be used, try using `theano.tensor.maximum(x, 0.)` as opposed to `theano.tensor.nnet.relu`.

# 2016-04-09

- Fixed bug of TNPG (max_backtracks should be set to 1 instead of 0) 
- Neural network policies now use tanh nonlinearities by default
- Refactored interface for `rllab.sampler.parallel_sampler`. Extracted new module `rllab.sampler.stateful_pool` containing general parallelization utilities.
- Fixed numerous issues in tests causing too long to run.
- Merged release branch onto master and removed the release branch, to avoid potential confusions.

# 2016-04-08

Features:
- Upgraded Mujoco interface to accomodate v1.30
