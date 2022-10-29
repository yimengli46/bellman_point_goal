import common
import os
import environments
import gridmap
import lsp
import matplotlib.pyplot as plt
import numpy as np
import shapely
import time
import torch
import vertexnav
import vertexnav_lsp


def _eval_single_strategy(args, world, unity_bridge, known_map, map_data,
                          start_pose, goal, do_plan_with_naive):
    chosen_subgoal = None
    verbose = True
    did_succeed = True

    # Helper function for creating a new robot instance
    robot = lsp.robot.Turtlebot_Robot(start_pose,
                                      primitive_length=args.step_size,
                                      num_primitives=args.num_primitives,
                                      map_data=map_data)
    prev_pose = None

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    estimator = vertexnav_lsp.models.VertexLSPOmni.get_net_eval_fn(
        args.network_file, device=device)

    simulator = lsp.simulators.Simulator(known_map,
                                         goal,
                                         args,
                                         unity_bridge=unity_bridge,
                                         world=world)
    simulator.frontier_grouping_inflation_radius = (
        simulator.inflation_radius)

    vertex_simulator = lsp.simulators.Simulator(known_map,
                                                goal,
                                                args,
                                                unity_bridge=unity_bridge,
                                                world=world)

    vlsp_planner = vertexnav_lsp.planners.LearnedVertexnavSubgoalPlanner(
        goal=goal, args=args, verbose=True)

    perfect_vertex_graph = vertexnav.vertex_graph.PerfectVertexGraph()
    vertex_graph = vertexnav.prob_vertex_graph.ProbVertexGraph()
    vertex_graph.DO_SLAM = True

    counter = 0
    count_since_last_turnaround = 100
    fn_start_time = time.time()

    # Main planning loop
    while (np.abs(robot.pose.x - goal.x) >= 3 * args.step_size
            or np.abs(robot.pose.y - goal.y) >= 3 * args.step_size):

        if verbose:
            print(f"Goal: {goal.x}, {goal.y}")
            print(f"Robot: {robot.pose.x}, {robot.pose.y} [motion: {robot.net_motion}]")
            print(f"Counter: {counter} | Count since last turnaround: "
                  f"{count_since_last_turnaround}")

        # Compute observations and update map
        pano_image, pano_depth_image = vertex_simulator.get_image(robot, do_get_depth=True)

        # Get the vertexnav data
        v_pose = vertexnav.Pose(robot.pose.x, robot.pose.y, robot.pose.yaw)

        # Update the perfect vertex graph
        perfect_obs = vertexnav.noisy.convert_world_obs_to_noisy_detection(
            world.get_vertices_for_pose(v_pose, max_range=args.laser_max_range_m),
            v_pose, do_add_noise=False, cov_rt=[[0.1**2, 0], [0, 0.1**2]])
        perfect_vertex_graph.add_observation(perfect_obs, v_pose)
        proposed_world = perfect_vertex_graph.get_proposed_world(
            num_detections_lower_limit=1)
        ksp = perfect_vertex_graph.get_known_poly(proposed_world, args.laser_max_range_m)
        print(f"Perfect area: {ksp.area}")

        # Query the network and update the learned map
        goal_loc_x_vec, goal_loc_y_vec = lsp.utils.learning_vision.get_rel_goal_loc_vecs(
            v_pose, goal, args.num_bearing
        )
        net_out_dict = estimator(image=pano_image / 255,
                                 goal_loc_x=goal_loc_x_vec,
                                 goal_loc_y=goal_loc_y_vec)

        noisy_observation = vertexnav.noisy.convert_net_grid_data_to_noisy_detection(
            net_out_dict,
            v_pose,
            max_range=args.max_range,
            num_range=args.num_range,
            num_bearing=args.num_bearing,
            sig_r=args.sig_r,
            sig_th=args.sig_th,
            nn_peak_thresh=args.nn_peak_thresh
        )

        if prev_pose is None:
            vertex_graph.add_observation(noisy_observation, r_pose=v_pose)
        else:
            odom = vertexnav.Pose.get_odom(p_new=v_pose, p_old=prev_pose)
            print(odom.x, odom.y, odom.yaw)
            vertex_graph.add_observation(noisy_observation, odom=odom)
            args.num_robots = 1
            if counter % 5 == 0:
                vertex_graph.sample_states(
                    vertex_association_time_window=15 * args.num_robots,
                    vertex_sampling_time_window=15 * args.num_robots,
                    num_topology_samples=20,
                    num_vertex_samples=50,
                    vertex_association_dist_threshold=20,
                    do_update_state=True)

        prev_pose = v_pose

        # Compute the visibility polygon
        last_pose_ind = len(vertex_graph.r_poses) - 1
        visibility_polygon = shapely.geometry.Polygon(
            vertexnav.noisy.compute_conservative_space_from_obs(
                vertex_graph.r_poses[last_pose_ind],
                vertex_graph.observations[last_pose_ind],
                args.laser_max_range_m,
            )).buffer(
                args.base_resolution, resolution=0, cap_style=3, join_style=2)

        # Compute vertex grid and new subgoals
        noisy_proposed_world = vertex_graph.get_proposed_world()
        ksp = vertex_graph.get_known_poly(noisy_proposed_world, args.laser_max_range_m)
        vertex_grid = world.get_grid_from_poly(ksp, noisy_proposed_world)
        inflated_grid = vertex_simulator.get_inflated_grid(vertex_grid, robot)
        inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
            inflated_grid, v_pose)
        new_subgoals = vertex_simulator.get_updated_frontier_set(inflated_grid, robot, set())
        print(f"Noisy area: {ksp.area}")

        # Update the learned planner
        vlsp_planner.update(
            observation=net_out_dict,
            observed_map=vertex_grid,
            subgoals=new_subgoals,
            robot_pose=vertex_graph.r_poses[last_pose_ind],
            visibility_polygon=visibility_polygon)

        # Add local obstacles before planning
        ranges = pano_depth_image[pano_depth_image.shape[0] // 2]
        directions, _ = vertexnav.utils.calc.directions_vec(ranges.size)
        vertex_grid = gridmap.mapping.insert_scan(vertex_grid,
                                                  directions,
                                                  laser_ranges=ranges,
                                                  max_range=6.0,
                                                  sensor_pose=robot.pose,
                                                  connect_neighbor_distance=2)
        inflated_grid = vertex_simulator.get_inflated_grid(vertex_grid, robot)

        if args.do_visualize_data:
            # Get the plotting grid
            plotting_grid = lsp.utils.plotting.make_plotting_grid(
                vertex_grid)
            for subgoal in vlsp_planner.subgoals:
                pf = subgoal.prob_feasible
                plotting_grid[subgoal.points[0, :],
                              subgoal.points[1, :], 0] = 1 - pf
                plotting_grid[subgoal.points[0, :],
                              subgoal.points[1, :], 1] = pf
                plotting_grid[subgoal.points[0, :],
                              subgoal.points[1, :], 2] = 0

            plt.figure()
            plt.subplot(3, 2, 1)
            plt.imshow(pano_image)
            plt.subplot(3, 2, 3)
            plt.imshow(net_out_dict['is_vertex'])
            plt.subplot(3, 2, 5)
            plt.imshow(net_out_dict['subgoal_prob_feasible'])
            plt.subplot(2, 2, 2)
            plt.imshow(vertex_grid.T)
            vertexnav.plotting.plot_proposed_world(plt.gca(),
                                                   proposed_world,
                                                   do_show_points=True,
                                                   do_plot_visibility=False,
                                                   robot_pose=v_pose)
            plt.subplot(2, 2, 4)
            plt.imshow(np.transpose(plotting_grid, (1, 0, 2)))
            vertexnav.plotting.plot_proposed_world(plt.gca(),
                                                   noisy_proposed_world,
                                                   do_show_points=True,
                                                   do_plot_visibility=False,
                                                   robot_pose=v_pose)
            plt.savefig(f"/data/dbg/vertexnav_lsp_{counter:04d}.png", dpi=150)
            # plt.show()

        if do_plan_with_naive:
            if verbose:
                print("Planning with naive/Dijkstra planner.")
            planning_grid = lsp.core.mask_grid_with_frontiers(
                inflated_grid,
                [],
            )
        else:
            chosen_subgoal = vlsp_planner.compute_selected_subgoal()
            if verbose:
                print("Planning via subgoal masking.")
            planning_grid = lsp.core.mask_grid_with_frontiers(
                inflated_grid, vlsp_planner.subgoals, do_not_mask=chosen_subgoal)

        # Check that the plan is feasible and compute path
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            planning_grid, [goal.x, goal.y], use_soft_cost=True)
        did_plan, path = get_path([robot.pose.x, robot.pose.y],
                                  do_sparsify=True, do_flip=True)

        # Move the robot
        motion_primitives = robot.get_motion_primitives()
        do_use_path = (count_since_last_turnaround > 10)
        costs, _ = lsp.primitive.get_motion_primitive_costs(
            planning_grid,
            cost_grid,
            robot.pose,
            path,
            motion_primitives,
            do_use_path=do_use_path)
        if abs(min(costs)) < 1e10:
            primitive_ind = np.argmin(costs)
            robot.move(motion_primitives, primitive_ind)
            if primitive_ind == len(motion_primitives) - 1:
                count_since_last_turnaround = -1
        else:
            # Force the robot to return to known space
            cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                planning_grid, [goal.x, goal.y],
                use_soft_cost=True,
                obstacle_cost=1e5)
            did_plan, path = get_path([robot.pose.x, robot.pose.y],
                                      do_sparsify=True,
                                      do_flip=True)
            costs, _ = lsp.primitive.get_motion_primitive_costs(
                planning_grid,
                cost_grid,
                robot.pose,
                path,
                motion_primitives,
                do_use_path=False)
            robot.move(motion_primitives, np.argmin(costs))

        # Check that the robot is not 'stuck'.
        if robot.max_travel_distance(
                num_recent_poses=100) < 5 * args.step_size:
            print("Planner stuck")
            did_succeed = False
            break

        if robot.net_motion > 4000 or counter > 600:
            print("Reached maximum distance.")
            did_succeed = False
            break

        counter += 1
        count_since_last_turnaround += 1
        if verbose:
            print("")

    if verbose:
        print("TOTAL TIME:", time.time() - fn_start_time)

    # One final round of sampling for plotting
    vertex_graph.sample_states(
        vertex_association_time_window=15 * args.num_robots,
        vertex_sampling_time_window=15 * args.num_robots,
        num_topology_samples=20,
        num_vertex_samples=50,
        vertex_association_dist_threshold=20,
        do_update_state=True)
    noisy_proposed_world = vertex_graph.get_proposed_world()
    ksp = vertex_graph.get_known_poly(noisy_proposed_world, args.laser_max_range_m)

    return {
        'did_succeed': did_succeed,
        'robot_grid': vertex_grid,
        'path': vertex_graph.r_poses,
        'perfect_path': perfect_vertex_graph.r_poses,
        'dist': common.compute_path_length(robot.all_poses),
        'proposed_world': noisy_proposed_world,
        'known_space_poly': ksp,
        'known_world': world,
    }


def eval_main(args):
    known_map, map_data, start_pose, goal = environments.generate.map_and_poses(args)

    if args.unity_path is None:
        raise ValueError('Unity Environment Required')

    # Write starting seed to the log file
    logfile = os.path.join(args.save_dir, args.logfile_name)
    with open(logfile, "a+") as f:
        f.write(f"LOG: {args.current_seed}\n")

    # Initialize the world and builder objects
    world = vertexnav.environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements)
    builder = environments.simulated.WorldBuildingUnityBridge

    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)

        # Run using the learned planner
        learned_out_data = _eval_single_strategy(args, world, unity_bridge, known_map, map_data,
                                                 start_pose, goal, do_plan_with_naive=False)

        # Run using the non-learned planner
        naive_out_data = _eval_single_strategy(args, world, unity_bridge, known_map, map_data,
                                               start_pose, goal, do_plan_with_naive=True)

    did_succeed = learned_out_data['did_succeed'] and naive_out_data['did_succeed']
    with open(logfile, "a+") as f:
        err_str = '' if did_succeed else '[ERR]'
        f.write(f"[Learn] {err_str} s: {args.current_seed:4d}"
                f" | learned: {learned_out_data['dist']:0.3f}"
                f" | baseline: {naive_out_data['dist']:0.3f}\n")

    # Write final plot to file
    image_file = os.path.join(
        args.save_dir,
        f"maze_learned_{args.current_seed}.png"
    )

    def _make_plot(out_data, name):
        # plt.imshow(
        #     lsp.utils.plotting.make_plotting_grid(out_data['robot_grid'], known_map))
        plt.gca().set_facecolor([0.8, 0.8, 0.8])
        final_pose = out_data['path'][len(out_data['path']) - 1]
        vertexnav.plotting.plot_polygon(plt.gca(),
                                        out_data['known_space_poly'],
                                        color=[1.0, 1.0, 1.0],
                                        alpha=1.0)
        vertexnav.plotting.plot_world(plt.gca(),
                                      out_data['known_world'],
                                      alpha=0.2)
        vertexnav.plotting.plot_proposed_world(plt.gca(),
                                               out_data['proposed_world'],
                                               do_show_points=True,
                                               do_plot_visibility=False,
                                               robot_pose=final_pose)
        xs = [p.x for p in out_data['perfect_path']]
        ys = [p.y for p in out_data['perfect_path']]
        plt.plot(xs, ys, '--', color='gray')
        xs = [p.x for p in out_data['path']]
        ys = [p.y for p in out_data['path']]
        plt.plot(xs, ys, 'r')
        plt.plot(final_pose.x, final_pose.y, 'go')
        plt.savefig(image_file, dpi=150)
        plt.title(f"{name} Cost: {out_data['dist']:.2f}")

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    _make_plot(learned_out_data, "Learned")
    plt.subplot(122)
    _make_plot(naive_out_data, "Naive")
    plt.savefig(image_file, dpi=150)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--xpassthrough', type=str, default='false')
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--data_file_base_name', type=str)
    parser.add_argument('--logfile_name', type=str, default='logfile.txt')
    parser.add_argument('--num_range',
                        type=int,
                        help='Number of range cells in output grid.')
    parser.add_argument('--num_bearing',
                        type=int,
                        help='Number of bearing cells in output grid.')
    parser.add_argument('--sig_r', type=float, default=10.0)
    parser.add_argument('--sig_th', type=float, default=0.25)
    parser.add_argument('--nn_peak_thresh', type=float, required=True)
    args = parser.parse_args()
    args.do_visualize_data = args.xpassthrough == 'true'

    args.max_range = args.laser_max_range_m / args.base_resolution

    eval_main(args)
