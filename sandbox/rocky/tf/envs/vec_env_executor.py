from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import cPickle as pickle
from sandbox.rocky.tf.misc import tensor_utils


class VecEnvExecutor(object):
    def __init__(self, env, n, max_path_length):
        envs = [pickle.loads(pickle.dumps(env)) for _ in xrange(n)]
        self.envs = envs
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_n):
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = map(list, zip(*all_results))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self):
        results = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass
