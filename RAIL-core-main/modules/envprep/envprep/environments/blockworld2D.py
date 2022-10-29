"""
Creates an instance/object/class of an 2D
environment for blockworld
(top down view) and its components
like initial state and goal state.
It also returns the PDDL problem
based on the environment, initial state
and goal state as well as other parameters.
"""

import itertools
import numpy as np
import pddlstream
import pddlstream.language.generator as pddl_generator
import pddlstream.language.constants as pddl_constants
from pddlstream.language.external import (
    defer_shared,
    get_defer_all_unbound,
    get_defer_any_unbound,
)
from pddlstream.language.function import FunctionInfo
from pddlstream.language.stream import StreamInfo
from examples.continuous_tamp.optimizer.optimizer import (
    cfree_motion_fn,
    get_optimize_fn,
)
from envprep.environments.blockworld.primitives import (
    get_pose_gen,
    collision_test,
    get_region_test,
    plan_motion,
    GRASP,
)
from examples.continuous_tamp.primitives import (
    distance_fn,
    duration_fn,
    inverse_kin_fn,
    MOVE_COST,
)


INITIAL_CONF = np.array([50, 250])


class BlockworldEnvironment:
    INITIAL_ROBOT_CONF = INITIAL_CONF

    def __init__(
        self,
        regions,
        obstacles,
        use_stream=True,
        use_optimizer=False,
        collisions=True,
        robot_return_to_start=True,
        verbose=False,
    ):
        self.regions = regions
        self.obstacles = obstacles
        self.verbose = verbose
        self.domain_pddl = pddlstream.utils.read(
            pddlstream.utils.get_file_path(__file__, "../domain.pddl")
        )
        self.constraints = pddlstream.algorithms.constraints.PlanConstraints(
            skeletons=None, exact=True, max_cost=pddl_constants.INF
        )

        self.robot_confs = {"r0": INITIAL_CONF}
        self.goal_confs = {"r0": INITIAL_CONF} if robot_return_to_start else None
        # Value handling
        external_paths = []
        if use_stream and use_optimizer:
            raise ValueError("Cannot have both use_stream=True and use_optimizer=True")
        elif not (use_stream or use_optimizer):
            raise ValueError("Must have one of use_stream=True or use_optimizer=True")
        if use_stream:
            external_paths.append(
                pddlstream.utils.get_file_path(__file__, "../stream.pddl")
            )
        if use_optimizer:
            external_paths.append(
                pddlstream.utils.get_file_path(__file__, "../optimizer/optimizer.pddl")
            )
        self.external_pddl = [pddlstream.utils.read(path) for path in external_paths]

        self.constant_map = {}
        self.stream_map = {
            "s-grasp": pddl_generator.from_fn(
                lambda b: pddlstream.language.constants.Output(GRASP)
            ),
            "s-region": pddl_generator.from_gen_fn(get_pose_gen(self.regions)),
            "s-ik": pddl_generator.from_fn(inverse_kin_fn),
            "s-motion": pddl_generator.from_fn(plan_motion),
            "t-region": pddl_generator.from_test(get_region_test(self.regions)),
            "t-cfree": pddl_generator.from_test(
                lambda *args: pddlstream.utils.implies(
                    collisions, not collision_test(*args)
                )
            ),
            "dist": distance_fn,
            "duration": duration_fn,  # temporal
        }
        if use_optimizer:
            # To avoid loading gurobi
            self.stream_map.update(
                {
                    "gurobi": pddl_generator.from_list_fn(
                        get_optimize_fn(self.regions, collisions=collisions)
                    ),
                    "rrt": pddl_generator.from_fn(cfree_motion_fn),
                }
            )

        defer_fn = defer_shared
        self.stream_info = {
            "s-region": StreamInfo(defer_fn=defer_fn),
            "s-grasp": StreamInfo(defer_fn=defer_fn),
            "s-ik": StreamInfo(defer_fn=get_defer_all_unbound(inputs="?g")),
            "s-motion": StreamInfo(defer_fn=get_defer_any_unbound()),
            "t-cfree": StreamInfo(
                defer_fn=get_defer_any_unbound(), eager=False, verbose=False
            ),
            "t-region": StreamInfo(eager=True, p_success=0),
            "dist": FunctionInfo(
                eager=False,
                defer_fn=get_defer_any_unbound(),
                opt_fn=lambda q1, q2: MOVE_COST,
            ),
            "gurobi-cfree": StreamInfo(eager=False, negate=True),
        }

    def get_pddl_problem(self, block_state, goal_conditions):
        initial_block_poses = block_state
        manipulate_cost = 1.0
        init = [
            pddl_constants.Equal(("Cost",), manipulate_cost),
            pddl_constants.Equal((pddl_constants.TOTAL_COST,), 0),
        ]

        init += [("Obstacle", self.obstacles)]

        for b in initial_block_poses.keys():
            for r in self.regions:
                init.append(("Placeable", b, r))

        for b, p in initial_block_poses.items():
            init += [
                ("Block", b),
                ("Pose", b, p),
                ("AtPose", b, p),
            ]

        goal_literals = []
        for b, r in goal_conditions:
            if isinstance(r, np.ndarray):
                init += [("Pose", b, r)]
                goal_literals += [("AtPose", b, r)]
            else:
                blocks = [b] if isinstance(b, str) else b
                regions = [r] if isinstance(r, str) else r
                conditions = []
                for body, region in itertools.product(blocks, regions):
                    init += [("Region", region), ("Placeable", body, region)]
                    conditions += [("In", body, region)]
                goal_literals.append(pddl_constants.Or(*conditions))

        for r, q in self.robot_confs.items():
            init += [
                ("Robot", r),
                ("CanMove", r),
                ("Conf", q),
                ("AtConf", r, q),
                ("HandEmpty", r),
            ]
        for r, q in self.goal_confs.items():
            goal_literals += [("HandEmpty", r)]
            goal_literals += [("AtConf", r, q)]

        pddl_problem = pddlstream.language.constants.PDDLProblem(
            self.domain_pddl,
            self.constant_map,
            self.external_pddl,
            self.stream_map,
            init,
            pddl_constants.And(*goal_literals),
        )

        if self.verbose:
            print("Initial:", pddlstream.utils.sorted_str_from_list(pddl_problem.init))
            print("Goal:", pddlstream.utils.str_from_object(pddl_problem.goal))

        return pddl_problem
