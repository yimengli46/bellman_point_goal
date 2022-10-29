import organization
import pytest


def test_organization_costs_move_single_block():
    regions = {"grey": (-10, 10), "red": (-10, -5), "green": (5, 10)}
    environment = organization.environments.blockworld.BlockworldEnvironment(
        regions=regions, verbose=True
    )

    cost_red_to_red = organization.core.get_cost_for_task(
        environment, block_state={"blockA": (-8, 0)}, task=[("blockA", "red")]
    )
    assert cost_red_to_red == pytest.approx(0)

    cost_green_to_red = organization.core.get_cost_for_task(
        environment, block_state={"blockA": (8, 0)}, task=[("blockA", "red")]
    )
    assert cost_red_to_red < cost_green_to_red

    cost_grey_to_red = organization.core.get_cost_for_task(
        environment, block_state={"blockA": (0, 0)}, task=[("blockA", "red")]
    )
    assert cost_grey_to_red < cost_green_to_red


def test_cost_for_two_blocks_no_crash():
    regions = {"grey": (-10, 10), "green": (-10, -5), "red": (5, 10)}
    environment = organization.environments.blockworld.BlockworldEnvironment(
        regions=regions, verbose=True
    )
    organization.core.get_cost_for_task(
        environment,
        block_state={"blockA": (0, 0), "blockB": (4, 0)},
        task=[("blockA", "green"), ("blockB", "green")],
    )


def test_organization_expected_costs_move_single_block():
    regions = {"grey": (-10, 10), "green": (-10, -5), "red": (5, 10)}
    environment = organization.environments.blockworld.BlockworldEnvironment(
        regions=regions, verbose=True
    )

    cost_grey_to_green = organization.core.get_cost_for_task(
        environment, block_state={"blockA": (3, 0)}, task=[("blockA", "green")]
    )
    cost_grey_to_red = organization.core.get_cost_for_task(
        environment, block_state={"blockA": (3, 0)}, task=[("blockA", "red")]
    )
    expected_cost_lr_1 = organization.core.get_expected_cost(
        environment,
        block_state={"blockA": (3, 0)},
        task_distribution=[[0.8, [("blockA", "green")]], [0.2, [("blockA", "red")]]],
    )
    PDDL_BUFFER = 5.0
    assert (
        abs(0.8 * cost_grey_to_green + 0.2 * cost_grey_to_red - expected_cost_lr_1)
        < PDDL_BUFFER
    )


def test_organization_costs_with_obstruction():
    regions = {"grey": (-10, 10), "red": (7, 10)}
    environment = organization.environments.blockworld.BlockworldEnvironment(
        regions=regions, verbose=True
    )

    cost_single_block = organization.core.get_cost_for_task(
        environment, block_state={"blockA": (0, 0)}, task=[("blockA", "red")]
    )

    cost_multi_block_no_obstruction = organization.core.get_cost_for_task(
        environment,
        block_state={"blockA": (0, 0), "blockB": (4, 0)},
        task=[("blockA", "red")],
    )
    PDDL_BUFFER = 5.0
    assert cost_single_block <= cost_multi_block_no_obstruction + PDDL_BUFFER

    cost_with_obstruction = organization.core.get_cost_for_task(
        environment,
        block_state={"blockA": (0, 0), "blockB": (8.5, 0)},
        task=[("blockA", "red")],
    )
    assert cost_multi_block_no_obstruction < cost_with_obstruction
