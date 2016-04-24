import numpy as np
from rllab.envs.box2d.parser import find_body

from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


# http://mlg.eng.cam.ac.uk/pilco/
class DoublePendulumEnv(Box2DEnv, Serializable):

    @autoargs.inherit(Box2DEnv.__init__)
    def __init__(self, *args, **kwargs):
        # make sure mdp-level step is 100ms long
        kwargs["frame_skip"] = kwargs.get("frame_skip", 2)
        if kwargs.get("template_args", {}).get("noise", False):
            self.link_len = (np.random.rand()-0.5) + 1
        else:
            self.link_len = 1
        kwargs["template_args"] = kwargs.get("template_args", {})
        kwargs["template_args"]["link_len"] = self.link_len
        super(DoublePendulumEnv, self).__init__(
            self.model_path("double_pendulum.xml.mako"),
            *args, **kwargs
        )
        self.link1 = find_body(self.world, "link1")
        self.link2 = find_body(self.world, "link2")
        Serializable.__init__(self, *args, **kwargs)

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        stds = np.array([0.1, 0.1, 0.01, 0.01])
        pos1, pos2, v1, v2 = np.random.randn(*stds.shape) * stds
        self.link1.angle = pos1
        self.link2.angle = pos2
        self.link1.angularVelocity = v1
        self.link2.angularVelocity = v2
        return self.get_current_obs()

    def get_tip_pos(self):
        cur_center_pos = self.link2.position
        cur_angle = self.link2.angle
        cur_pos = (
            cur_center_pos[0] - self.link_len*np.sin(cur_angle),
            cur_center_pos[1] - self.link_len*np.cos(cur_angle)
        )
        return cur_pos

    @overrides
    def compute_reward(self, action):
        yield
        tgt_pos = np.asarray([0, self.link_len * 2])
        cur_pos = self.get_tip_pos()
        dist = np.linalg.norm(cur_pos - tgt_pos)
        yield -dist

    def is_current_done(self):
        return False

