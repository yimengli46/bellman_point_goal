import pytest  # noqa: F401
import envprep
import organization
from envprep.utilities.gcn_data_gen_helper import (
    get_all_regions_for_blocks,
    get_block_states,
)
from envprep.environments.blockworld.primitives import (
    interval_contains,
    get_block_interval,
)


def test_prepared_state_three_regions():
    regions = {
        "blue": [(0, 300), (100, 400)],
        "red": [(200, 300), (300, 400)],
        "green": [(300, 100), (400, 200)],
    }
    # initial_block_state = {"A": np.array([362.11001894, 134.08810558]), "B": np.array([250.15553979, 350.93839801])}
    blocks_poses = get_all_regions_for_blocks(regions, 2)
    task_distribution = [[0.8, [("A", "red")]], [0.2, [("A", "blue")]]]
    environment = envprep.environments.blockworld2D.BlockworldEnvironment(
        regions=regions, obstacles=[], verbose=True
    )
    prepared_state = []
    lowest_exp_cost = float("inf")
    for i in range(len(blocks_poses)):
        block_state = get_block_states(blocks_poses[i])
        expected_cost_state = organization.core.get_expected_cost(
            environment, block_state, task_distribution
        )
        if lowest_exp_cost > expected_cost_state:
            lowest_exp_cost = expected_cost_state
            prepared_state = list(block_state.values())

    assert interval_contains(
        regions.get("green"), get_block_interval("b2", prepared_state[1])
    )
