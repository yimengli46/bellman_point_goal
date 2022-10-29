import lsp
import lsp_cond
import environments
import numpy as np
import pytest
import gridmap
import torch
import random
import itertools
from lsp_cond.core import RolloutState
from lsp.core import FState
import time as time
import matplotlib.pyplot as plt
from matplotlib import cm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
BLANK_IMAGE = np.zeros((128, 512, 3), dtype=np.uint8)
viridis = cm.get_cmap('viridis')


def _get_env_and_args():
    parser = lsp.utils.command_line.get_parser()
    args = parser.parse_args(['--save_dir', ''])
    args.current_seed = 10008  # 1037
    args.map_type = 'maze'
    args.step_size = 1.8
    args.num_primitives = 16
    args.field_of_view_deg = 360
    args.base_resolution = 1.0
    args.inflation_radius_m = 2.5
    args.laser_max_range_m = 120
    args.num_range = 32
    args.num_bearing = 128
    args.autoencoder_network_file = '/data/lsp_conditional/logs/gstein/AutoEncoder.pt'
    args.network_file = '/data/lsp_conditional/logs/gstein/model.pt'
    args.experiment_name = 'test'
    args.image_file = '/data/lsp_conditional/test/img'

    # Create the map
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    # Comment the line below when not willing start from the middle of the path
    pose = lsp_cond.utils.get_path_middle_point(known_map, pose, goal, args)
    return known_map, map_data, pose, goal, args


def run_planning_loop(known_map, map_data, pose, goal, args, unity_path,
                      planners, num_steps=None, do_plan_with_naive=False, 
                      do_yield_planner=False, do_plot=False):
    # Initialize the world and builder objects
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements)
    builder = environments.simulated.WorldBuildingUnityBridge
    robot = lsp.robot.Turtlebot_Robot(pose,
                                      primitive_length=args.step_size,
                                      num_primitives=args.num_primitives,
                                      map_data=map_data)
    if not isinstance(planners, list):
        planners = [planners]
    with builder(unity_path) as unity_bridge:
        unity_bridge.make_world(world)

        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=world)
        simulator.frontier_grouping_inflation_radius = (
            simulator.inflation_radius)

        planning_loop = lsp.planners.PlanningLoop(goal,
                                                  known_map,
                                                  simulator,
                                                  unity_bridge,
                                                  robot,
                                                  args,
                                                  verbose=False)

        for counter, step_data in enumerate(planning_loop):
            for planner in planners:
                # Update the planner objects
                start_time = time.time()
                planner.update(
                    {'image': step_data['image']},
                    step_data['robot_grid'],
                    step_data['subgoals'],
                    step_data['robot_pose'])
                print('Time taken to update:', time.time() - start_time)
            chosen_subgoal = planners[0].compute_selected_subgoal()

            if do_plot:  # Controls whether to plot the planning graph
                if chosen_subgoal is not None:
                    planning_grid = lsp.core.mask_grid_with_frontiers(
                        planners[0].inflated_grid, planners[0].subgoals, do_not_mask=chosen_subgoal)
                else:
                    planning_grid = lsp.core.mask_grid_with_frontiers(
                        planners[0].inflated_grid,
                        [],
                    )
                # Check that the plan is feasible and compute path
                cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                    planning_grid, [goal.x, goal.y], use_soft_cost=True)
                did_plan, path = get_path([robot.pose.x, robot.pose.y],
                                          do_sparsify=True,
                                          do_flip=True)

                # Plotting
                plt.ion()
                plt.figure(1)
                plt.clf()
                ax = plt.subplot(211)
                plt.imshow(step_data['image'])
                ax = plt.subplot(212)
                lsp_cond.plotting.plot_pose(ax, robot.pose, color='blue')
                lsp_cond.plotting.plot_grid_with_frontiers(
                    ax, step_data['robot_grid'], known_map, step_data['subgoals'])
                lsp_cond.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_cond.plotting.plot_path(ax, path)
                lsp_cond.plotting.plot_pose_path(ax, robot.all_poses)
                for (s, e) in planners[0].edge_data:
                    ps = planners[0].current_graph[s][e]['pts']
                    plt.plot(ps[:, 0], ps[:, 1], 'lightgray')
                
                is_subgoal = planners[0].is_subgoal
                prob_feasible = planners[0].out
                for vp_idx, ps in enumerate(planners[0].vertex_points):
                    color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
                    plt.plot(ps[0], ps[1], '.', color=color, markersize=4)
                image_file = args.image_file + str(counter) + '.png'
                plt.savefig(image_file, dpi=150)

            if do_yield_planner:
                if len(planners) == 1:
                    yield planners[0]
                else:
                    yield planners

            if planners is not None and not do_plan_with_naive:
                planning_loop.set_chosen_subgoal(chosen_subgoal)

            if num_steps is not None and counter >= num_steps:
                break


def test_lsp_cond_cnn_output_test():
    torch.manual_seed(8616)
    np.random.seed(8616)
    known_map, map_data, pose, goal, args = _get_env_and_args()
    device = DEVICE
    model = lsp_cond.learning.models.auto_encoder.AutoEncoder()
    model.load_state_dict(torch.load(args.autoencoder_network_file, 
                                     map_location=DEVICE)) 
    model.eval()
    model.to(device)

    image = []
    goal_loc_x = []
    goal_loc_y = []
    subgoal_loc_x = []
    subgoal_loc_y = []

    for _ in range(2):
        RANDOM_IMAGE = np.random.rand(128, 512, 3) * 255
        image.append(RANDOM_IMAGE)

    for _ in range(2):
        BLANK_IMAGE = np.zeros((128, 512, 3))
        image.append(BLANK_IMAGE)

    num_of_nodes = 4
    goal_loc_x = np.random.rand(num_of_nodes, 1, 128)
    goal_loc_x[2:] = 0
    goal_loc_y = np.random.rand(num_of_nodes, 1, 128)
    goal_loc_y[2:] = 0
    subgoal_loc_x = np.random.rand(num_of_nodes, 1, 128)
    subgoal_loc_x[2:] = 0
    subgoal_loc_y = np.random.rand(num_of_nodes, 1, 128)
    subgoal_loc_y[2:] = 0

    datum = {
        'image': image,
        'goal_loc_x': goal_loc_x,
        'goal_loc_y': goal_loc_y,
        'subgoal_loc_x': subgoal_loc_x,
        'subgoal_loc_y': subgoal_loc_y,
    }
    data = lsp_cond.utils.preprocess_cnn_data(datum)

    with torch.no_grad():
        latent_features = model.encoder(data, device).detach().cpu().numpy()
    np.testing.assert_equal(
        np.any(np.not_equal(
            latent_features[0], latent_features[1])), True)
    np.testing.assert_allclose(
        latent_features[2], latent_features[3], rtol=1e-06)
    np.testing.assert_equal(
        np.any(np.not_equal(
            latent_features[0], latent_features[2])), True)


def test_lsp_cond_latent_feature_calculation_compuation_saver(do_debug_plot, unity_path):
    """ Test whether the output is consistent when computation saving approach
    is applied with the one where the latent feature is calculated for all the 
    nodes.
    """
    known_map, map_data, pose, goal, args = _get_env_and_args()
    split_planner = lsp_cond.planners. \
        ConditionalUnknownSubgoalPlanner(goal=goal, args=args, device=DEVICE)
    combined_planner = lsp_cond.planners. \
        ConditionalCombinedPlanner(goal=goal, args=args, device=DEVICE)
    planners = [split_planner, combined_planner]
    for split_planner, combined_planner in run_planning_loop(
            known_map, map_data, pose, 
            goal, args, unity_path,
            planners, 
            num_steps=30,
            do_plan_with_naive=False,
            do_yield_planner=True):
        np.testing.assert_allclose(
            split_planner.out, combined_planner.out, rtol=1e-05)


def test_lsp_cond_plan_loop_unknown(do_debug_plot, unity_path):
    """Confirm that planning with "no subgoals" does not crash."""
    known_map, map_data, pose, goal, args = _get_env_and_args()
    planner = lsp_cond.planners.ConditionalUnknownSubgoalPlanner(goal=goal, 
                                                                 args=args,
                                                                 device=DEVICE)
    for planner in run_planning_loop(
            known_map, map_data, pose, goal, args, 
            unity_path, planner, num_steps=10, 
            do_plan_with_naive=False, do_plot=False):
        pass


def test_lsp_cond_plan_loop_known(do_debug_plot, unity_path):
    """Confirm that planning with "no subgoals" does not crash."""
    known_map, map_data, pose, goal, args = _get_env_and_args()
    planner = lsp_cond.planners.ConditionalKnownSubgoalPlanner(goal=goal, 
                                                               args=args, 
                                                               known_map=known_map)
    run_planning_loop(known_map, map_data, pose, goal, args, unity_path,
                      planner, num_steps=20, do_plan_with_naive=True)


def test_lsp_cond_all_node_inputs(do_debug_plot, unity_path):
    """ Tests observations for each node
    """ 
    known_map, map_data, pose, goal, args = _get_env_and_args()
    planner = lsp_cond.planners.ConditionalUnknownSubgoalPlanner(
        goal=goal, 
        args=args,
        device=DEVICE
    )
    last_dict_keys = []
    for planner in run_planning_loop(
        known_map, map_data, pose, goal, args, unity_path, planner, 
            num_steps=5, do_plan_with_naive=False, do_yield_planner=True):
        assert planner.cnn_input['image']
        assert planner.gcn_graph_input['is_subgoal']

        total_pixel_values = [
            data['image'].sum()
            for data in planner.new_node_dict.values()
        ]

        new_dict_keys = list(planner.new_node_dict.keys())
        vp = [tuple(vertex_point) for vertex_point in planner.vertex_points]
        # Check if the nn input dictionary has entry for every vertex point 
        assert new_dict_keys == vp

        # If any new_dict_key is not in last_dict_keys then all the vertices
        # are new and this is the first step so all images should be the
        # same otherwise if new_dict_keys is not the same as old_dict_keys
        # then there must be new vertices with new observations
        if last_dict_keys == []:
            assert np.std(total_pixel_values) == pytest.approx(0)
        elif new_dict_keys != last_dict_keys:
            assert np.std(total_pixel_values) > 0
        elif new_dict_keys == last_dict_keys:  # Unnecessary
            print("same!")
        last_dict_keys = list(planner.new_node_dict.keys())


def test_lsp_cond_ensure_node_for_each_subgoal(do_debug_plot, unity_path):
    """ Test to ensure that each and every subgoal node gets associated with
    only distinct vertex point
    """
    known_map, map_data, pose, goal, args = _get_env_and_args()
    planner = lsp_cond.planners.ConditionalUnknownSubgoalPlanner(
        goal=goal, 
        args=args,
        device=DEVICE
    )
    for planner in run_planning_loop(
        known_map, map_data, pose, goal, args, unity_path, planner, 
            num_steps=50, do_plan_with_naive=False, do_yield_planner=True,
            do_plot=False):
        assert len(planner.subgoals) == len(planner.subgoal_nodes.keys())


def test_lsp_cond_plan_loop_datum_correct_properties(do_debug_plot, unity_path):
    """Show that we can get datum for the new subgoals."""
    known_map, map_data, pose, goal, args = _get_env_and_args()
    planner = lsp_cond.planners.ConditionalKnownSubgoalPlanner(goal=goal, 
                                                               args=args, 
                                                               known_map=known_map)
    for planner in run_planning_loop(known_map, map_data, pose, goal, args,
                                     unity_path, planner, num_steps=10,
                                     do_plan_with_naive=True,
                                     do_yield_planner=True):
        training_data = planner.compute_training_data()
        expected_keys = [
            'image', 'goal_loc_x', 'goal_loc_y', 'subgoal_loc_x',
            'subgoal_loc_y', 'is_subgoal', 'history',
            'edge_data', 'is_feasible', 'delta_success_cost',
            'exploration_cost', 'positive_weighting', 'negative_weighting'
        ]
        training_data_keys = list(training_data.keys())
        for key in expected_keys:
            assert key in training_data_keys


def _get_fake_data_old_format(num_subgoals):
    """Generate fake subgoal data in the old LSP format."""
    random.seed(8616)

    cost_exp = 100
    cost_ds = 30
    dist_goal = 200
    dist_robot = 20
    dist_frontier = 50

    # Make a bunch of random data
    subgoals = [
        lsp.core.Frontier(np.array([[ii, ii]]).T) for ii in range(num_subgoals)
    ]
    [
        s.set_props(prob_feasible=random.random(),
                    delta_success_cost=cost_exp * random.random(),
                    exploration_cost=cost_ds * random.random())
        for s in subgoals
    ]

    goal_distances = {s: dist_goal * random.random() for s in subgoals}
    robot_distances = {s: dist_robot * random.random() for s in subgoals}
    frontier_distances = {
        frozenset(pair): dist_frontier * random.random()
        for pair in itertools.combinations(subgoals, 2)
    }

    distances = {
        'goal': goal_distances,
        'robot': robot_distances,
        'frontier': frontier_distances
    }
    
    return {'distances': distances,
            'subgoals': subgoals}


def convert_old_data_to_marginal_data(subgoals, old_distances, 
                                      make_conditional=False):
    # Split into two functions to convert the subgoal data and 
    # the distanes data for the new API
    random.seed(8616)
    marginal_subgoal_props = np.array([[subgoal.prob_feasible, 
                                      subgoal.delta_success_cost, 
                                      subgoal.exploration_cost]  # Nx3 
                                      for subgoal in subgoals])
    is_subgoal = [1] * len(subgoals)
    rollout_histories = lsp_cond.utils. \
        generate_all_rollout_history(is_subgoal)

    if make_conditional:
        new_data = {tuple(history): np.copy(marginal_subgoal_props) * random.random()
                    for history in rollout_histories}
    else:
        new_data = {tuple(history): np.copy(marginal_subgoal_props)
                    for history in rollout_histories}
    subgoal_ind_dict = {subgoal: ii for ii, subgoal in enumerate(subgoals)}

    new_distances = {}
    new_distances['goal'] = {
        subgoal_ind_dict[subgoal]: goal_dist
        for subgoal, goal_dist in old_distances['goal'].items()
    }
    new_distances['robot'] = {
        subgoal_ind_dict[subgoal]: robot_dist
        for subgoal, robot_dist in old_distances['robot'].items()
    }
    new_distances['frontier'] = {
        frozenset([subgoal_ind_dict[s] for s in old_key]): subgoal_dist
        for old_key, subgoal_dist in old_distances['frontier'].items()}
    return new_data, new_distances


@pytest.mark.parametrize("subgoal_ordering", ([1, 2, 0], [2, 3, 1, 0], [0, 1, 2, 3, 4, 5]))
def test_lsp_cond_marginal_matches_history_conditioned_travel_costs_given_subgoal_order(
    subgoal_ordering
):
    """Confirm that we can get a dictionary with the history-conditioned 
    properties and run some key functions without them crashing."""
    # Get some fake data (assert properties about it)
    number_of_subgoals = len(subgoal_ordering)
    old_data_dict = _get_fake_data_old_format(number_of_subgoals)
    state = None
    for subgoal in subgoal_ordering:
        state = FState(
            old_data_dict['subgoals'][subgoal],
            old_data_dict['distances'],
            old_state=state
        )

    # use a function to convert the old data format to new format 
    new_data, new_distances = convert_old_data_to_marginal_data(
        subgoals=old_data_dict['subgoals'], 
        old_distances=old_data_dict['distances']
    )
    
    rollout_state = None  # Set the initial state to be None
    for subgoal_idx in subgoal_ordering:
        rollout_state = RolloutState(
            new_data,
            new_distances,
            subgoal_idx, 
            old_state=rollout_state
        )

    assert rollout_state.cost == pytest.approx(state.cost)


def test_lsp_cond_changing_props_cost():
    subgoal_ordering = [0, 1, 2]
    number_of_subgoals = len(subgoal_ordering)
    old_data_dict = _get_fake_data_old_format(number_of_subgoals)
    new_data, new_distances = convert_old_data_to_marginal_data(
        subgoals=old_data_dict['subgoals'], 
        old_distances=old_data_dict['distances']
    )
    new_data[(1, 1, 1)][0] = [.4, 3, 1]
    new_data[(0, 1, 1)][1] = [.5, 5, 1]
    new_data[(0, 0, 1)][2] = [.6, 4, 2]

    rollout_state = None  # Set the initial state to be None
    for subgoal_idx in subgoal_ordering:
        rollout_state = RolloutState(
            new_data,
            new_distances,
            subgoal_idx, 
            old_state=rollout_state
        )
    assert 75.415055504 == pytest.approx(rollout_state.cost)
    assert .12 == pytest.approx(rollout_state.prob)


def test_lsp_cond_ordering_output_slow_and_moderately_fast():
    """ This "slow version" gives you the min ordering that you'll be
        comparing against. Your "faster version" should give you the same result.
    """
    number_of_subgoals = 4
    old_data_dict = _get_fake_data_old_format(number_of_subgoals)
    new_data, new_distances = convert_old_data_to_marginal_data(
        subgoals=old_data_dict['subgoals'], 
        old_distances=old_data_dict['distances'],
        make_conditional=True
    )
    start_time = time.time()
    ordering_cost_dict = {}
    subgoals = np.arange(number_of_subgoals)
    for subgoal_ordering in itertools.permutations(subgoals):
        rollout_state = None  # Set the initial state to be None
        for subgoal_idx in subgoal_ordering:
            rollout_state = RolloutState(
                subgoal_props=new_data,
                distances=new_distances,
                frontier_idx=subgoal_idx, 
                old_state=rollout_state)
        ordering_cost_dict[subgoal_ordering] = rollout_state.cost

    slow_min_ordering = min(ordering_cost_dict, key=ordering_cost_dict.get)
    slow_min_cost = ordering_cost_dict[slow_min_ordering]
    print('Slow method timing:', time.time() - start_time)
    start_time = time.time()
    fast_min_cost, fast_min_ordering = lsp_cond.core.get_lowest_cost_ordering(
        new_data,
        new_distances
    )
    print('Faster method timing:', time.time() - start_time)
    assert fast_min_cost == pytest.approx(slow_min_cost)
    assert fast_min_ordering == pytest.approx(slow_min_ordering)


def test_lsp_cond_rollout_cost():
    subgoal_ordering = [0, 1, 2]
    number_of_subgoals = len(subgoal_ordering)
    old_data_dict = _get_fake_data_old_format(number_of_subgoals)
    new_data, new_distances = convert_old_data_to_marginal_data(
        subgoals=old_data_dict['subgoals'], 
        old_distances=old_data_dict['distances'],
        make_conditional=True
    )
    rollout_state = None  # Set the initial state to be None
    for subgoal_idx in subgoal_ordering:
        rollout_state = RolloutState(
            new_data,
            new_distances,
            subgoal_idx, 
            old_state=rollout_state
        )
        print(rollout_state.cost, rollout_state.history, rollout_state.frontier_list)
    assert 182.09783746619829 == pytest.approx(rollout_state.cost)
    assert 0.09042769080722117 == pytest.approx(rollout_state.prob)
