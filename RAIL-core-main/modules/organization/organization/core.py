import pddlstream
import pddlstream.algorithms.meta
import pddlstream.language.constants as pddl_constants

DEFAULT_CONFIG = None


def get_cost_for_task(environment, block_state, task):
    """Short summary.

    Parameters
    ----------
    environment : environment where task is being executed on.
    block_state : blocks and their initial state in the environment.
    task: task to execute in the envionent.

    Returns
    -------
    cost: cost of completing a task with PDDLstream

    """
    pddl_problem = environment.get_pddl_problem(block_state, task)

    hierarchy = []
    replan_actions = set()

    # success_cost = 0 if args.optimal else INF
    success_cost = pddl_constants.INF
    planner = "max-astar"
    effort_weight = 1.0 / pddlstream.algorithms.downward.get_cost_scale()
    solution = pddlstream.algorithms.meta.solve(
        pddl_problem,
        algorithm="adaptive",
        constraints=environment.constraints,
        stream_info=environment.stream_info,
        replan_actions=replan_actions,
        planner=planner,
        max_planner_time=pddl_constants.INF,
        hierarchy=hierarchy,
        max_time=pddl_constants.INF,
        max_iterations=pddl_constants.INF,
        debug=False,
        verbose=environment.verbose,
        unit_costs=False,
        success_cost=success_cost,
        unit_efforts=False,
        effort_weight=effort_weight,
        search_sample_ratio=1,
        visualize=False,
    )

    if environment.verbose:
        pddl_constants.print_solution(solution)

    # plan, cost, evaluations = solution
    plan, cost, _ = solution

    if plan is None:
        return None
    if any(p.name == "move" and p.args[2] is None for p in plan):
        return None

    return cost


def get_expected_cost(environment, block_state, task_distribution):
    """Short summary.

    Parameters
    ----------
    environment :  environment where task is being executed on.
    block_state : blocks and their initial state in the environment.
    task_distribution : distribution over tasks in the environment.

    Returns
    -------
    expected_cost: expected cost of an state given task distribution

    """
    expected_costs = []
    for prob, task in task_distribution:
        cost = get_cost_for_task(environment, block_state, task)
        if cost is None:
            return None

        exp_cost = prob * cost
        expected_costs.append(exp_cost)

    return sum(expected_costs)
