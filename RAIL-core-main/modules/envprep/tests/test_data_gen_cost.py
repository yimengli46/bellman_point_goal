import pytest  # noqa: F401
import organization
import numpy as np
import envprep


def test_none_expected_cost():
    regions = {
        "grey": [(0, 0), (400, 400)],
        "blue": [(300, 0), (400, 100)],
        "red": [(0, 0), (100, 100)],
        "yellow": [(200, 0), (300, 100)],
    }

    obs1 = [
        [(0, 195), (150, 205)],
        [(195, 0), (205, 150)],
        [(195, 250), (205, 400)],
        [(250, 195), (400, 205)],
    ]
    obs2 = [[(0, 150), (150, 160)]]

    block_state = {"A": np.array([350, 50]), "B": np.array([50, 50])}
    env1 = envprep.environments.blockworld2D.BlockworldEnvironment(
        regions=regions, obstacles=obs1, verbose=True
    )
    env2 = envprep.environments.blockworld2D.BlockworldEnvironment(
        regions=regions, obstacles=obs2, verbose=True
    )
    task_distribution = [[1, [("A", "red")]]]
    exp_cost_none = organization.core.get_expected_cost(
        env1, block_state, task_distribution
    )
    exp_cost = organization.core.get_expected_cost(env2, block_state, task_distribution)
    assert exp_cost_none is None
    assert exp_cost is not None
