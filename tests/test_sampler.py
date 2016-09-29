

import numpy as np


def test_truncate_paths():
    from rllab.sampler.parallel_sampler import truncate_paths

    paths = [
        dict(
            observations=np.zeros((100, 1)),
            actions=np.zeros((100, 1)),
            rewards=np.zeros(100),
            env_infos=dict(),
            agent_infos=dict(lala=np.zeros(100)),
        ),
        dict(
            observations=np.zeros((50, 1)),
            actions=np.zeros((50, 1)),
            rewards=np.zeros(50),
            env_infos=dict(),
            agent_infos=dict(lala=np.zeros(50)),
        ),
    ]

    truncated = truncate_paths(paths, 130)
    assert len(truncated) == 2
    assert len(truncated[-1]["observations"]) == 30
    assert len(truncated[0]["observations"]) == 100
    # make sure not to change the original one
    assert len(paths) == 2
    assert len(paths[-1]["observations"]) == 50
