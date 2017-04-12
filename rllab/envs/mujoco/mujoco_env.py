import numpy as np
import os.path as osp
from cached_property import cached_property

from rllab import spaces
from rllab.envs.base import Env
from rllab.misc.overrides import overrides
from rllab.mujoco_py import MjModel, MjViewer
from rllab.misc import autoargs
from rllab.misc import logger
import theano
import tempfile
import os
import mako.template
import mako.lookup

MODEL_DIR = osp.abspath(
    osp.join(
        osp.dirname(__file__),
        '../../../vendor/mujoco_models'
    )
)

BIG = 1e6

def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]

def q_mult(a, b): # multiply two quaternion
    w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    i = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    j = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    k = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    return [w, i, j, k]

class MujocoEnv(Env):
    FILE = None

    @autoargs.arg('action_noise', type=float,
                  help='Noise added to the controls, which will be '
                       'proportional to the action bounds')
    def __init__(self, action_noise=0.0, file_path=None, template_args=None):
        # compile template
        if file_path is None:
            if self.__class__.FILE is None:
                raise "Mujoco file not specified"
            file_path = osp.join(MODEL_DIR, self.__class__.FILE)
        if file_path.endswith(".mako"):
            lookup = mako.lookup.TemplateLookup(directories=[MODEL_DIR])
            with open(file_path) as template_file:
                template = mako.template.Template(
                    template_file.read(), lookup=lookup)
            content = template.render(
                opts=template_args if template_args is not None else {},
            )
            tmp_f, file_path = tempfile.mkstemp(text=True)
            with open(file_path, 'w') as f:
                f.write(content)
            self.model = MjModel(file_path)
            os.close(tmp_f)
        else:
            self.model = MjModel(file_path)
        self.data = self.model.data
        self.viewer = None
        self.init_qpos = self.model.data.qpos
        self.init_qvel = self.model.data.qvel
        self.init_qacc = self.model.data.qacc
        self.init_ctrl = self.model.data.ctrl
        self.qpos_dim = self.init_qpos.size
        self.qvel_dim = self.init_qvel.size
        self.ctrl_dim = self.init_ctrl.size
        self.action_noise = action_noise
        if "frame_skip" in self.model.numeric_names:
            frame_skip_id = self.model.numeric_names.index("frame_skip")
            addr = self.model.numeric_adr.flat[frame_skip_id]
            self.frame_skip = int(self.model.numeric_data.flat[addr])
        else:
            self.frame_skip = 1
        if "init_qpos" in self.model.numeric_names:
            init_qpos_id = self.model.numeric_names.index("init_qpos")
            addr = self.model.numeric_adr.flat[init_qpos_id]
            size = self.model.numeric_size.flat[init_qpos_id]
            init_qpos = self.model.numeric_data.flat[addr:addr + size]
            self.init_qpos = init_qpos
        self.dcom = None
        self.current_com = None
        self.reset()
        super(MujocoEnv, self).__init__()

    @cached_property
    @overrides
    def action_space(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return spaces.Box(lb, ub)

    @cached_property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @property
    def action_bounds(self):
        return self.action_space.bounds

    def reset_mujoco(self, init_state=None):
        if init_state is None:
            self.model.data.qpos = self.init_qpos + \
                                   np.random.normal(size=self.init_qpos.shape) * 0.01
            self.model.data.qvel = self.init_qvel + \
                                   np.random.normal(size=self.init_qvel.shape) * 0.1
            self.model.data.qacc = self.init_qacc
            self.model.data.ctrl = self.init_ctrl
        else:
            start = 0
            for datum_name in ["qpos", "qvel", "qacc", "ctrl"]:
                datum = getattr(self.model.data, datum_name)
                datum_dim = datum.shape[0]
                datum = init_state[start: start + datum_dim]
                setattr(self.model.data, datum_name, datum)
                start += datum_dim

    @overrides
    def reset(self, init_state=None):
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def get_current_obs(self):
        return self._get_full_obs()

    def _get_full_obs(self):
        data = self.model.data
        cdists = np.copy(self.model.geom_margin).flat
        for c in self.model.data.contact:
            cdists[c.geom2] = min(cdists[c.geom2], c.dist)
        obs = np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            # data.cdof.flat,
            data.cinert.flat,
            data.cvel.flat,
            # data.cacc.flat,
            data.qfrc_actuator.flat,
            data.cfrc_ext.flat,
            data.qfrc_constraint.flat,
            cdists,
            # data.qfrc_bias.flat,
            # data.qfrc_passive.flat,
            self.dcom.flat,
        ])
        return obs

    @property
    def _state(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

    @property
    def _full_state(self):
        return np.concatenate([
            self.model.data.qpos,
            self.model.data.qvel,
            self.model.data.qacc,
            self.model.data.ctrl,
        ]).ravel()

    def inject_action_noise(self, action):
        # generate action noise
        noise = self.action_noise * \
                np.random.normal(size=action.shape)
        # rescale the noise to make it proportional to the action bounds
        lb, ub = self.action_bounds
        noise = 0.5 * (ub - lb) * noise
        return action + noise

    def forward_dynamics(self, action):
        self.model.data.ctrl = self.inject_action_noise(action)
        for _ in range(self.frame_skip):
            self.model.step()
        self.model.forward()
        new_com = self.model.data.com_subtree[0]
        self.dcom = new_com - self.current_com
        self.current_com = new_com

    def get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer

    def render(self, close=False, mode='human'):
        if mode == 'human':
            viewer = self.get_viewer()
            viewer.loop_once()
        elif mode == 'rgb_array':
            viewer = self.get_viewer()
            viewer.loop_once()
            # self.get_viewer(config=config).render()
            data, width, height = self.get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        if close:
            self.stop_viewer()

    def start_viewer(self):
        viewer = self.get_viewer()
        if not viewer.running:
            viewer.start()

    def stop_viewer(self):
        if self.viewer:
            self.viewer.finish()

    def release(self):
        # temporarily alleviate the issue (but still some leak)
        from rllab.mujoco_py.mjlib import mjlib
        mjlib.mj_deleteModel(self.model._wrapped)
        mjlib.mj_deleteData(self.data._wrapped)

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.body_comvels[idx]

    def print_stats(self):
        super(MujocoEnv, self).print_stats()
        print("qpos dim:\t%d" % len(self.model.data.qpos))

    def action_from_key(self, key):
        raise NotImplementedError
