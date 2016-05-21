import theano.tensor as TT
import numpy as np
import theano
from rllab.distributions.categorical import Categorical
from rllab.distributions.base import Distribution

TINY = 1e-8


class RecurrentCategorical(Distribution):
    def __init__(self, dim):
        self._cat = Categorical(dim)
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of two categorical distributions
        """
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        # Assume layout is N * T * A
        return TT.sum(
            old_prob_var * (TT.log(old_prob_var + TINY) - TT.log(new_prob_var + TINY)),
            axis=2
        )

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two categorical distributions
        """
        old_prob = old_dist_info["prob"]
        new_prob = new_dist_info["prob"]
        return np.sum(
            old_prob * (np.log(old_prob + TINY) - np.log(new_prob + TINY)),
            axis=2
        )

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        # Assume layout is N * T * A
        a_dim = x_var.shape[-1]
        flat_ratios = self._cat.likelihood_ratio_sym(
            x_var.reshape((-1, a_dim)),
            dict(prob=old_prob_var.reshape((-1, a_dim))),
            dict(prob=new_prob_var.reshape((-1, a_dim)))
        )
        return flat_ratios.reshape(old_prob_var.shape[:2])

    def entropy(self, dist_info):
        probs = dist_info["prob"]
        return -np.sum(probs * np.log(probs + TINY), axis=2)

    def log_likelihood_sym(self, xs, dist_info_vars):
        probs = dist_info_vars["prob"]
        # Assume layout is N * T * A
        a_dim = probs.shape[-1]
        # a_dim = TT.printing.Print("lala")(a_dim)
        flat_logli = self._cat.log_likelihood_sym(xs.reshape((-1, a_dim)), dict(prob=probs.reshape((-1, a_dim))))
        return flat_logli.reshape(probs.shape[:2])

    def log_likelihood(self, xs, dist_info):
        probs = dist_info["prob"]
        # Assume layout is N * T * A
        a_dim = probs.shape[-1]
        flat_logli = self._cat.log_likelihood_sym(xs.reshape((-1, a_dim)), dict(prob=probs.reshape((-1, a_dim))))
        return flat_logli.reshape(probs.shape[:2])

    @property
    def dist_info_keys(self):
        return ["prob"]
