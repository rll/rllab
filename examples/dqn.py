from __future__ import print_function 
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import theano
import tenano.tensor as TT
from lasagne.updates import adam

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,))

N = 100
T = 100

