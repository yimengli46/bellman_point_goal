#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import time
from itertools import product
import pytest  # noqa: F401
from pddlstream.algorithms.meta import solve
from envprep.environments.blockworld.primitives import (
    get_pose_gen,
    collision_test,
    get_region_test,
    plan_motion,
    SUCTION_WIDTH,
    GRASP,
    ENVIRONMENT_NAMES,
    TAMPProblem,
    interval_contains,
    get_block_interval,
)
from examples.continuous_tamp.primitives import (
    distance_fn,
    duration_fn,
    inverse_kin_fn,
    MOVE_COST,
    update_state,
    TAMPState,
)
from pddlstream.algorithms.downward import get_cost_scale
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.language.external import (
    defer_shared,
    get_defer_all_unbound,
    get_defer_any_unbound,
)
from pddlstream.language.constants import (
    And,
    Equal,
    PDDLProblem,
    TOTAL_COST,
    Or,
    Output,
)
from pddlstream.language.function import FunctionInfo
from pddlstream.language.generator import from_gen_fn, from_test, from_fn
from pddlstream.language.stream import StreamInfo
from pddlstream.language.temporal import get_end, compute_duration, retime_plan
from pddlstream.utils import (
    user_input,
    read,
    INF,
    get_file_path,
    implies,
    inclusive_range,
    Profiler,
)


##################################################


def _create_problem(tamp_problem, hand_empty=False, manipulate_cost=1.0):
    initial = tamp_problem.initial
    assert not initial.holding
    init = [Equal(("Cost",), manipulate_cost), Equal((TOTAL_COST,), 0)]
    init += [
        ("Placeable", b, r)
        for b in initial.block_poses.keys()
        for r in tamp_problem.regions
        if (r in ENVIRONMENT_NAMES)
    ]

    init += [("Obstacle", tamp_problem.obstacles)]

    for b, p in initial.block_poses.items():
        init += [("Block", b), ("Pose", b, p), ("AtPose", b, p)]

    goal_literals = []
    for b, r in tamp_problem.goal_regions.items():
        if isinstance(r, np.ndarray):
            init += [("Pose", b, r)]
            goal_literals += [("AtPose", b, r)]
        else:
            blocks = [b] if isinstance(b, str) else b
            regions = [r] if isinstance(r, str) else r
            conditions = []
            for body, region in product(blocks, regions):
                init += [("Region", region), ("Placeable", body, region)]
                conditions += [("In", body, region)]
            goal_literals.append(Or(*conditions))

    for r, q in initial.robot_confs.items():
        init += [
            ("Robot", r),
            ("CanMove", r),
            ("Conf", q),
            ("AtConf", r, q),
            ("HandEmpty", r),
        ]
        if hand_empty:
            goal_literals += [("HandEmpty", r)]
        if tamp_problem.goal_conf is not None:
            # goal_literals += [('AtConf', tamp_problem.goal_conf)]
            goal_literals += [("AtConf", r, q)]

    goal = And(*goal_literals)

    return init, goal


def _pddlstream_from_tamp(tamp_problem, use_stream=True, collisions=True):

    domain_pddl = read(get_file_path(__file__, "../envprep/domain.pddl"))
    external_paths = []
    if use_stream:
        external_paths.append(get_file_path(__file__, "../envprep/stream.pddl"))
    external_pddl = [read(path) for path in external_paths]

    constant_map = {}
    stream_map = {
        "s-grasp": from_fn(lambda b: Output(GRASP)),
        "s-region": from_gen_fn(get_pose_gen(tamp_problem.regions)),
        "s-ik": from_fn(inverse_kin_fn),
        "s-motion": from_fn(plan_motion),
        "t-region": from_test(get_region_test(tamp_problem.regions)),
        "t-cfree": from_test(
            lambda *args: implies(collisions, not collision_test(*args))
        ),
        "dist": distance_fn,
        "duration": duration_fn,
    }

    init, goal = _create_problem(tamp_problem)
    return PDDLProblem(domain_pddl, constant_map, external_pddl, stream_map, init, goal)


##################################################


def _display_plan(tamp_problem, plan, display=True, time_step=0.08, sec_per_step=1e-3):
    from envprep.environments.blockworld.viewer import ContinuousTMPViewer
    from examples.discrete_tamp.viewer import COLORS

    colors = dict(zip(sorted(tamp_problem.initial.block_poses.keys()), COLORS))
    viewer = ContinuousTMPViewer(
        SUCTION_WIDTH,
        tamp_problem.regions,
        tamp_problem.obstacles,
        title="Continuous TAMP",
    )
    state = tamp_problem.initial
    duration = compute_duration(plan)

    viewer.draw_state(state, colors)
    if display:
        user_input("Start?")
    if plan is not None:
        for t in inclusive_range(0, duration, time_step):
            for action in plan:
                if action.start <= t <= get_end(action):
                    update_state(state, action, t - action.start)
            viewer.draw_state(state, colors)
            if display:
                if sec_per_step is None:
                    user_input("Continue?")
                else:
                    time.sleep(sec_per_step)
    if display:
        user_input("Finish?")

    return state


defer_fn = defer_shared
skeletons = None
max_cost = INF
constraints = PlanConstraints(skeletons=skeletons, exact=True, max_cost=max_cost)
replan_actions = set()
stream_info = {
    "s-region": StreamInfo(defer_fn=defer_fn),
    "s-grasp": StreamInfo(defer_fn=defer_fn),
    "s-ik": StreamInfo(defer_fn=get_defer_all_unbound(inputs="?g")),
    "s-motion": StreamInfo(defer_fn=get_defer_any_unbound()),
    "t-cfree": StreamInfo(defer_fn=get_defer_any_unbound(), eager=False, verbose=False),
    "t-region": StreamInfo(eager=True, p_success=0),
    "dist": FunctionInfo(
        eager=False, defer_fn=get_defer_any_unbound(), opt_fn=lambda q1, q2: MOVE_COST
    ),
    "gurobi-cfree": StreamInfo(eager=False, negate=True),
}

success_cost = INF
planner = "max-astar"
effort_weight = 1.0 / get_cost_scale()


def _get_tamp_problem(obstacles=[], goal_regions={"A": "red"}):
    block_poses = {"A": (350, 50), "B": (50, 50)}
    regions = {
        "grey": [(0, 0), (400, 400)],
        "blue": [(300, 0), (400, 100)],
        "red": [(0, 0), (100, 100)],
        "white": [(200, 300), (300, 400)],
    }
    tamp_problem = TAMPProblem(
        initial=TAMPState(
            robot_confs={"r0": np.array([200, 200])},
            holding={},
            block_poses=block_poses,
        ),
        regions=regions,
        obstacles=obstacles,
        goal_conf=np.array([200, 200]),
        goal_regions=goal_regions,
    )

    return tamp_problem


def _get_solution(pddlstream_problem):
    with Profiler(field="cumtime", num=20):
        solution = solve(
            pddlstream_problem,
            algorithm="adaptive",
            constraints=constraints,
            stream_info=stream_info,
            replan_actions=replan_actions,
            planner=planner,
            max_planner_time=1000,
            hierarchy=[],
            max_time=100,
            max_iterations=INF,
            debug=False,
            verbose=True,
            unit_costs=False,
            success_cost=success_cost,
            unit_efforts=True,
            effort_weight=effort_weight,
            search_sample_ratio=1,
            visualize=False,
        )
    return solution


def _assert_func(do_debug_plot, tamp_problem, goal=[(0, 0), (100, 100)]):
    pddlstream_problem = _pddlstream_from_tamp(tamp_problem, use_stream=True)
    solution = _get_solution(pddlstream_problem)
    plan, _, _ = solution
    if do_debug_plot:
        if plan is not None:
            _display_plan(tamp_problem, retime_plan(plan))
    assert plan is not None
    places = []
    for p in plan:
        if p.name == "move":
            assert p.args[2] is not None
        if p.name == "place" and p.args[1] == "A":
            places.append(p)

    pos = places[-1].args[2]
    assert interval_contains(goal, get_block_interval("b", pos))


def test_2d_blockworld_unblocked(do_debug_plot):
    diff_goal = [(200, 300), (300, 400)]
    tamp_problem = _get_tamp_problem(goal_regions={"A": "white"})
    _assert_func(do_debug_plot, tamp_problem, goal=diff_goal)


def test_2d_blockworld_blocked(do_debug_plot):
    tamp_problem = _get_tamp_problem()
    _assert_func(do_debug_plot, tamp_problem)


def test_2d_blockworld_with_one_obstacle(do_debug_plot):
    one_obstacle = [[(0, 150), (150, 160)]]
    tamp_problem = _get_tamp_problem(obstacles=one_obstacle)
    _assert_func(do_debug_plot, tamp_problem)


def test_2d_blockworld_with_two_obstacles(do_debug_plot):
    two_obstacles = [[(0, 150), (150, 160)], [(250, 0), (255, 200)]]
    tamp_problem = _get_tamp_problem(obstacles=two_obstacles)
    _assert_func(do_debug_plot, tamp_problem)
