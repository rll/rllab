import numpy as np

from rllab.envs.dm_control_env import DmControlEnv
from rllab.envs.normalized_env import normalize

from dm_control import suite


def run_task(domain_name, task_name):
    print("run: domain %s task %s" % (domain_name, task_name))
    dmcontrol_env = normalize(
        DmControlEnv(
            domain_name=domain_name,
            task_name=task_name,
            plot=True,
            width=600,
            height=400),
        normalize_obs=False,
        normalize_reward=False)

    time_step = dmcontrol_env.reset()
    action_spec = dmcontrol_env.action_space
    for _ in range(200):
        dmcontrol_env.render()
        action = action_spec.sample()
        next_obs, reward, done, info = dmcontrol_env.step(action)
        if done == True:
            break

    dmcontrol_env.terminate()


for domain, task in suite.ALL_TASKS:
    run_task(domain, task)

print("Congratulation! All tasks are done!")
