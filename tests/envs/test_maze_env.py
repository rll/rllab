import math

from rllab.envs.mujoco.maze.maze_env_utils import line_intersect, ray_segment_intersect


def test_line_intersect():
    assert line_intersect((0, 0), (0, 1), (0, 0), (1, 0))[:2] == (0, 0)
    assert line_intersect((0, 0), (0, 1), (0, 0), (0, 1))[2] == 0
    assert ray_segment_intersect(ray=((0, 0), 0), segment=((1, -1), (1, 1))) == (1, 0)
    assert ray_segment_intersect(ray=((0, 0), math.pi), segment=((1, -1), (1, 1))) is None
