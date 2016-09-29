import theano.tensor as TT
import numpy as np
from .base import Distribution
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

TINY = 1e-8


# def from_onehot_sym(x_var):
#     ret = TT.zeros((x_var.shape[0],), x_var.dtype)
#     nonzero_n, nonzero_a = TT.nonzero(x_var)[:2]
#     ret = TT.set_subtensor(ret[nonzero_n], nonzero_a.astype('uint8'))
#     return ret


def from_onehot(x_var):
    ret = np.zeros((len(x_var),), 'int32')
    nonzero_n, nonzero_a = np.nonzero(x_var)
    ret[nonzero_n] = nonzero_a
    return ret


class Categorical(Distribution):
    def __init__(self, dim):
        self._dim = dim
        self._srng = RandomStreams()

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of two categorical distributions
        """
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        # Assume layout is N * A
        return TT.sum(
            old_prob_var * (TT.log(old_prob_var + TINY) - TT.log(new_prob_var + TINY)),
            axis=-1
        )

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two categorical distributions
        """
        old_prob = old_dist_info["prob"]
        new_prob = new_dist_info["prob"]
        return np.sum(
            old_prob * (np.log(old_prob + TINY) - np.log(new_prob + TINY)),
            axis=-1
        )

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        x_var = TT.cast(x_var, 'float32')
        # Assume layout is N * A
        return (TT.sum(new_prob_var * x_var, axis=-1) + TINY) / (TT.sum(old_prob_var * x_var, axis=-1) + TINY)

    def entropy(self, info):
        probs = info["prob"]
        return -np.sum(probs * np.log(probs + TINY), axis=1)

    def entropy_sym(self, dist_info_vars):
        prob_var = dist_info_vars["prob"]
        return -TT.sum(prob_var * TT.log(prob_var + TINY), axis=1)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        probs = dist_info_vars["prob"]
        # Assume layout is N * A
        return TT.log(TT.sum(probs * TT.cast(x_var, 'float32'), axis=-1) + TINY)

    def log_likelihood(self, xs, dist_info):
        probs = dist_info["prob"]
        # Assume layout is N * A
        N = probs.shape[0]
        return np.log(probs[np.arange(N), from_onehot(np.asarray(xs))] + TINY)

    def sample_sym(self, dist_info):
        probs = dist_info["prob"]
        return self._srng.multinomial(pvals=probs, dtype='uint8')

    @property
    def dist_info_keys(self):
        return ["prob"]
