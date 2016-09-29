from rllab.distributions.base import Distribution

class Delta(Distribution):
    @property
    def dim(self):
        return 0

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        return None

    def kl(self, old_dist_info, new_dist_info):
        return None

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        raise NotImplementedError

    def entropy(self, dist_info):
        raise NotImplementedError

    def log_likelihood_sym(self, x_var, dist_info_vars):
        raise NotImplementedError

    def likelihood_sym(self, x_var, dist_info_vars):
        return TT.exp(self.log_likelihood_sym(x_var, dist_info_vars))

    def log_likelihood(self, xs, dist_info):
        return None

    @property
    def dist_info_keys(self):
        return None

    def entropy(self,dist_info):
        return 0
