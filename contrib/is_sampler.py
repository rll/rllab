from rllab.algos.batch_polopt import BatchSampler
from math import exp
import random
import copy

class ISSampler(BatchSampler):
    """
    Sampler which alternates between live sampling iterations using BatchSampler
    and importance sampling iterations. Currently works with vpg, not guaranteed
    to work with other policy gradient methods: npo, ppo, trpo, etc.
    """
    def __init__(self, algo, n_backtrack = 'all'):
        """
        :type algo: BatchPolopt
        :param n_backtrack: Number of past policies to update from
        """
        self.n_backtrack = n_backtrack
        self._hist = []
        self._is_itr = 0

        super(ISSampler, self).__init__(algo)

    @property
    def history(self):
        """
        History of policies that have interacted with the environment and the
        data from interaction episode(s)
        """
        return self._hist

    def add_history(self, policy_distribution, paths):
        """
        Store policy distribution and paths in history
        """
        self._hist.append((policy_distribution, paths))

    def get_history_list(self, n_past = 'all'):
        """
        Get list of (distribution, data) tuples from history
        """
        if n_past == 'all':
            return self._hist
        return self._hist[-min(n_past, len(self._hist)):]

    def obtain_samples(self, itr):
        # Alternate between importance sampling and live sampling
        # This may mess with interpreting the data in the logs
        if self._is_itr:
            paths = self.obtain_is_samples(itr)
        else:
            paths = super(ISSampler, self).obtain_samples(itr)
            self.add_history(self.algo.policy.distribution, paths)

        self._is_itr = (self._is_itr + 1) % 2
        return paths


    def obtain_is_samples(self, itr):
        paths = []
        for hist_policy_distribution, hist_paths in self.get_history_list(self.n_backtrack):
            h_paths = self.sample_isweighted_paths(
                policy=self.algo.policy,
                hist_policy_distribution=hist_policy_distribution,
                max_samples=self.algo.batch_size,
                max_path_length=self.algo.max_path_length,
                paths=hist_paths,
            )
            paths.extend(h_paths)
        if len(paths) > self.algo.batch_size:
            paths = random.sample(paths, self.algo.batch_size)
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated

    def sample_isweighted_paths(
            self,
            policy,
            hist_policy_distribution,
            max_samples,
            max_path_length=100,
            paths=None,
            randomize_draw=True,
            ):

        if not paths:
            return []

        n_paths = len(paths)

        n_samples = min(len(paths), max_samples)

        if randomize_draw:
            samples = random.sample(paths, n_samples)
        elif paths:
            samples = paths[0:n_samples]

        # make duplicate of samples so we don't permanently alter historical data
        samples = copy.deepcopy(samples)

        dist1 = policy.distribution
        dist2 = hist_policy_distribution
        for path in samples:
            _, agent_infos = policy.get_actions(path['observations'])
            hist_agent_infos = path['agent_infos']
            path['agent_infos'] = agent_infos

            # apply importance sampling weight
            loglike_p = dist1.log_likelihood(path['actions'], agent_infos)
            loglike_hp = dist2.log_likelihood(path['actions'], hist_agent_infos)
            is_ratio = exp(sum(loglike_p) - sum(loglike_hp))
            path['rewards'] *= is_ratio

        return paths
