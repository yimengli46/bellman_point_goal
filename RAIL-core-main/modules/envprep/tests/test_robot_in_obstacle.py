import pytest  # noqa: F401
from envprep.environments.blockworld.primitives import get_robot_interval, contains


def test_obstacle():
    robot_center = [70, 70]
    robot_box_interval = get_robot_interval(robot_center)
    large_obstacle = [[0, 0], [300, 300]]
    right_obstacle = [[80, 0], [150, 300]]
    left_obstacle = [[20, 0], [50, 300]]
    top_obstacle = [[0, 20], [300, 50]]
    bottom_obstacle = [[0, 80], [300, 150]]
    small_obstacle = [[50, 0], [60, 200]]
    not_interfering_obstacle = [[10, 200], [20, 250]]

    assert contains(robot_box_interval, large_obstacle)
    assert contains(robot_box_interval, small_obstacle)
    assert contains(robot_box_interval, right_obstacle)
    assert contains(robot_box_interval, left_obstacle)
    assert contains(robot_box_interval, top_obstacle)
    assert contains(robot_box_interval, bottom_obstacle)
    assert not contains(robot_box_interval, not_interfering_obstacle)
