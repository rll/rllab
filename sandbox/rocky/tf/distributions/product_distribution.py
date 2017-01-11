

from .base import Distribution
import numpy as np
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.distributions.categorical import Categorical


class ProductDistribution(Distribution):
    def __init__(self, distributions):
        self.distributions = distributions
        self.dimensions = [x.dim for x in self.distributions]
        self._dim = sum(self.dimensions) 

    @property
    def dim(self):
        return self._dim

    def _split_x(self, x):
        """
        Split the tensor variable or value into per component.
        """
        cum_dims = list(np.cumsum(self.dimensions))
        out = []
        for slice_from, slice_to, dist in zip([0] + cum_dims, cum_dims, self.distributions):
            sliced = x[:, slice_from:slice_to]
            if isinstance(dist, Categorical):
                if isinstance(sliced, np.ndarray):
                    sliced = np.cast['uint8'](sliced)
                else:
                    sliced = tf.cast(sliced, dtype=tf.uint8)
            out.append(sliced)
        return out

    def _split_dist_info(self, dist_info):
        """
        Split the dist info dictionary into per component.
        """
        ret = []
        for idx, dist in enumerate(self.distributions):
            cur_dist_info = dict()
            for k in dist.dist_info_keys:
                cur_dist_info[k] = dist_info["id_%d_%s" % (idx, k)]
            ret.append(cur_dist_info)
        return ret

    def log_likelihood(self, xs, dist_infos):
        splitted_xs = self._split_x(xs)
        dist_infos = self._split_dist_info(dist_infos)
        ret = 0
        for x_i, dist_info_i, dist_i in zip(splitted_xs, dist_infos, self.distributions):
            ret += dist_i.log_likelihood(x_i, dist_info_i)
        return ret

    def log_likelihood_sym(self, x_var, dist_info_vars):
        splitted_x_vars = self._split_x(x_var)
        dist_info_vars = self._split_dist_info(dist_info_vars)
        ret = 0
        for x_var_i, dist_info_var_i, dist_i in zip(splitted_x_vars, dist_info_vars, self.distributions):
            ret += dist_i.log_likelihood_sym(x_var_i, dist_info_var_i)
        return ret

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return tf.exp(logli_new - logli_old)

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_dist_info_vars = self._split_dist_info(old_dist_info_vars)
        new_dist_info_vars = self._split_dist_info(new_dist_info_vars)
        ret = 0
        for old_dist_info_var_i, new_dist_info_var_i, dist_i in zip(
                old_dist_info_vars, new_dist_info_vars, self.distributions):
            ret += dist_i.kl_sym(old_dist_info_var_i, new_dist_info_var_i)
        return ret

    def kl(self, old_dist_infos, new_dist_infos):
        old_dist_infos = self._split_dist_info(old_dist_infos)
        new_dist_infos = self._split_dist_info(new_dist_infos)
        ret = 0
        for old_dist_info_i, new_dist_info_i, dist_i in zip(
                old_dist_infos, new_dist_infos, self.distributions):
            ret += dist_i.kl(old_dist_info_i, new_dist_info_i)
        return ret

    def entropy(self, info):
        dist_infos = self._split_dist_info(info)
        return np.sum([dist.entropy(info) for dist, info in zip(self.distributions, dist_infos)])

    @property
    def dist_info_specs(self):
        ret = []
        for idx, dist in enumerate(self.distributions):
            for k, dim in dist.dist_info_specs:
                ret.append(("id_%d_%s" % (idx, k), dim))
        return ret
    
    def sample(self, dist_info):
        specs = self._split_dist_info(dist_info)
        samples = [dist.sample(sp) for sp, dist in zip(specs, self.distributions)]
        
        assert len([samples[0]]) == 1 # only sample a single val
        
        return samples

    def sample_sym(self, dist_info):
        specs = self._split_dist_info(dist_info)
        samples = [dist.sample_sym(sp) for sp, dist in zip(specs, self.distributions)]
        return samples
        #samples = tf.pack(samples)
        #return tf.transpose(samples, [1, 0])
        
