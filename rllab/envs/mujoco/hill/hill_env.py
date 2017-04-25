import tempfile
import os
import time

import mako.template
import mako.lookup

from rllab.envs.proxy_env import ProxyEnv
from rllab.core.serializable import Serializable
import rllab.envs.mujoco.mujoco_env as mujoco_env
import rllab.envs.mujoco.hill.terrain as terrain
from rllab.misc import logger

MODEL_DIR = mujoco_env.MODEL_DIR

class HillEnv(ProxyEnv, Serializable):
    
    HFIELD_FNAME = 'hills.png'
    TEXTURE_FNAME = 'hills_texture.png'
    MIN_DIFFICULTY = 0.05
    
    def __init__(self,
                 difficulty=1.0,
                 texturedir='/tmp/mujoco_textures',
                 hfield_dir='/tmp/mujoco_terrains',
                 regen_terrain=True,
                 *args, **kwargs):
        Serializable.quick_init(self, locals())
        
        self.difficulty = max(difficulty, self.MIN_DIFFICULTY)
        self.texturedir = texturedir
        self.hfield_dir = hfield_dir
        
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        
        template_file_name = 'hill_' + model_cls.__module__.split('.')[-1] + '.xml.mako'

        template_options=dict(
            difficulty=self.difficulty,
            texturedir=self.texturedir,
            hfield_file=os.path.join(self.hfield_dir, self.HFIELD_FNAME))
        
        file_path = os.path.join(MODEL_DIR, template_file_name)
        lookup = mako.lookup.TemplateLookup(directories=[MODEL_DIR])
        with open(file_path) as template_file:
            template = mako.template.Template(
                template_file.read(), lookup=lookup)
        content = template.render(opts=template_options)
                
        tmp_f, file_path = tempfile.mkstemp(text=True)
        with open(file_path, 'w') as f:
            f.write(content)
        
        if self._iam_terrain_generator(regen_terrain):
            self._gen_terrain(regen_terrain)
            os.remove(self._get_lock_path())
            
        inner_env = model_cls(*args, file_path=file_path, **kwargs)  # file to the robot specifications
        ProxyEnv.__init__(self, inner_env)  # here is where the robot env will be initialized
    
        os.close(tmp_f)
    
    def _get_lock_path(self):
        return os.path.join(self.hfield_dir, '.lock')
    
    def _iam_terrain_generator(self, regen):
        ''' When parallel processing, don't want each worker to generate its own terrain. This method ensures that
        one worker generates the terrain, which is then used by other workers.
        It's still possible to have each worker use their own terrain by passing each worker a different hfield and
        texture dir.
        '''
        if not os.path.exists(self.hfield_dir):
            os.makedirs(self.hfield_dir)
        terrain_path = os.path.join(self.hfield_dir, self.HFIELD_FNAME)
        lock_path = self._get_lock_path()
        if regen or (not regen and not os.path.exists(terrain_path)):
            # use a simple lock file to prevent different workers overwriting the file, and/or running their own unique terrains
            if not os.path.exists(lock_path):
                with open(lock_path, 'w') as f:
                    f.write(str(os.getpid()))
                return True
            else:
                # wait for the worker that's generating the terrain to finish
                total = 0
                logger.log("Process {0} waiting for terrain generation...".format(os.getpid()))
                while os.path.exists(lock_path) and total < 120:
                    time.sleep(5)
                    total += 5
                if os.path.exists(lock_path):
                    raise "Process {0} timed out waiting for terrain generation, or stale lock file".format(os.getpid())
                logger.log("Done.")
                return False
            
    def _gen_terrain(self, regen=True):
        logger.log("Process {0} generating terrain...".format(os.getpid()))
        x, y, hfield = terrain.generate_hills(40, 40, 500)
        hfield = self._mod_hfield(hfield)
        terrain.save_heightfield(x, y, hfield, self.HFIELD_FNAME, path=self.hfield_dir)
        terrain.save_texture(x, y, hfield, self.TEXTURE_FNAME, path=self.texturedir)
        logger.log("Generated.")
            
    def _mod_hfield(self, hfield):
        '''Subclasses can override this to modify hfield'''
        return hfield
    
    def get_current_obs(self):
        return self._wrapped_env.get_current_obs()
