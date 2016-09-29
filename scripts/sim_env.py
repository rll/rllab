import argparse
import sys
import time

import numpy as np
import pygame

from rllab.envs.base import Env
# from rllab.env.base import MDP
from rllab.misc.resolve import load_class


def sample_action(lb, ub):
    Du = len(lb)
    if np.any(np.isinf(lb)) or np.any(np.isinf(ub)):
        raise ValueError('Cannot sample unbounded actions')
    return np.random.rand(Du) * (ub - lb) + lb


def to_onehot(ind, dim):
    ret = np.zeros(dim)
    ret[ind] = 1
    return ret


def visualize_env(env, mode, max_steps=sys.maxsize, speedup=1):
    timestep = 0.05
    # step ahead with all-zero action
    if mode == 'noop':
        for _ in range(max_steps):
            env.render()
            time.sleep(timestep / speedup)
    elif mode == 'random':
        env.reset()
        env.render()
        for i in range(max_steps):
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            # if i % 10 == 0:
            env.render()
            # import time as ttime
            time.sleep(timestep / speedup)
            if done:
                env.reset()
    elif mode == 'static':
        env.reset()
        while True:
            env.render()
            time.sleep(timestep / speedup)
    elif mode == 'human':
        if hasattr(env, 'start_interactive'):
            env.start_interactive()
        else:
            env.reset()
            env.render()
            tr = 0.
            from rllab.envs.box2d.box2d_env import Box2DEnv
            if isinstance(env, Box2DEnv):
                for _ in range(max_steps):
                    pygame.event.pump()
                    keys = pygame.key.get_pressed()
                    action = env.action_from_keys(keys)
                    ob, r, done, _ = env.step(action)
                    tr += r
                    env.render()
                    time.sleep(timestep / speedup)
                    if done:
                        tr = 0.
                        env.reset()
                return

            from rllab.envs.mujoco.mujoco_env import MujocoEnv
            from rllab.envs.mujoco.maze.maze_env import MazeEnv
            if isinstance(env, (MujocoEnv, MazeEnv)):
                trs = [tr]
                actions = [np.zeros(2)]
                from rllab.mujoco_py import glfw

                def cb(window, key, scancode, action, mods):
                    actions[0] = env.action_from_key(key)

                glfw.set_key_callback(env.viewer.window, cb)
                while True:
                    try:
                        actions[0] = np.zeros(2)
                        glfw.poll_events()
                        # if np.linalg.norm(actions[0]) > 0:
                        ob, r, done, info = env.step(actions[0])
                        trs[0] += r
                        env.render()
                        # time.sleep(env.timestep / speedup)
                        time.sleep(env.timestep / speedup)
                        if done:
                            trs[0] = 0.
                            env.reset()
                    except Exception as e:
                        print(e)
                return

            assert hasattr(env, "start_interactive"), "The environment must implement method start_interactive"

            env.start_interactive()
        # Assume using matplotlib
        # TODO - make this logic more legit

        # env.render()
        # import matplotlib.pyplot as plt
        # def handle_key_pressed(event):
        #     action = env.action_from_key(event.key)
        #     if action is not None:
        #         _, _, done, _ = env.step(action)
        #         if done:
        #             plt.close()
        #             return
        #         env.render()
        #
        # env.matplotlib_figure.canvas.mpl_connect('key_press_event', handle_key_pressed)
        # plt.ioff()
        # plt.show()

    else:
        raise ValueError('Unsupported mode: %s' % mode)
        # env.stop_viewer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
                        help='module path to the env class')
    parser.add_argument('--mode', type=str, default='static',
                        choices=['noop', 'random', 'static', 'human'],
                        help='module path to the env class')
    parser.add_argument('--speedup', type=float, default=1, help='speedup')
    parser.add_argument('--max_steps', type=int,
                        default=sys.maxsize, help='max steps')
    args = parser.parse_args()
    env = load_class(args.env, Env, ["rllab", "envs"])()
    visualize_env(env, mode=args.mode, max_steps=args.max_steps,
                  speedup=args.speedup)
