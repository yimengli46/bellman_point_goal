import os
import environments
import gridmap
import learning
import lsp
import matplotlib.pyplot as plt
import vertexnav
import vertexnav_lsp


def write_datum_to_pickle(args, counter, datum):
    save_dir = os.path.join(args.save_dir)
    data_filename = os.path.join('data', f'dat_{args.current_seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(
        os.path.join(save_dir, data_filename), datum)

    csv_filename = f'{args.data_file_base_name}_{args.current_seed}.csv'
    with open(os.path.join(save_dir, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')


def data_gen_main(args, do_plan_with_naive=True):
    known_map, map_data, start_pose, goal = environments.generate.map_and_poses(args)

    # Open the connection to Unity (if desired)
    if args.unity_path is None:
        raise ValueError('Unity Environment Required')

    # Initialize the world and builder objects
    world = vertexnav.environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements)
    builder = environments.simulated.WorldBuildingUnityBridge

    # Helper function for creating a new robot instance
    robot = lsp.robot.Turtlebot_Robot(start_pose,
                                      primitive_length=args.step_size,
                                      num_primitives=args.num_primitives,
                                      map_data=map_data)

    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)

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

        known_planner = lsp.planners.KnownSubgoalPlanner(
            goal=goal, known_map=known_map, args=args,
            do_compute_weightings=True)

        planning_loop = lsp.planners.PlanningLoop(goal,
                                                  known_map,
                                                  simulator,
                                                  unity_bridge,
                                                  robot,
                                                  args,
                                                  verbose=True)

        perfect_vertex_graph = vertexnav.vertex_graph.PerfectVertexGraph()

        for counter, step_data in enumerate(planning_loop):

            # Get the vertexnav data
            rpose = step_data['robot_pose']
            v_pose = vertexnav.Pose(rpose.x, rpose.y, rpose.yaw)
            datum = vertexnav.learning.get_vertex_datum_for_pose(
                v_pose,
                world,
                unity_bridge,
                min_range=-1,
                max_range=args.max_range,
                num_range=args.num_range,
                num_bearing=args.num_bearing)

            # Update the map
            perfect_obs = vertexnav.noisy.convert_world_obs_to_noisy_detection(
                world.get_vertices_for_pose(v_pose, max_range=args.laser_max_range_m),
                v_pose,
                do_add_noise=False,
                cov_rt=[[0.1**2, 0], [0, 0.1**2]])
            perfect_vertex_graph.add_observation(perfect_obs, v_pose)

            # Compute vertex grid and new subgoals
            proposed_world = perfect_vertex_graph.get_proposed_world(
                num_detections_lower_limit=1)
            ksp = perfect_vertex_graph.get_known_poly(proposed_world, args.laser_max_range_m)
            vertex_grid = world.get_grid_from_poly(ksp, proposed_world)
            inflated_grid = vertex_simulator.get_inflated_grid(vertex_grid, robot)
            inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
                inflated_grid, v_pose)
            new_subgoals = vertex_simulator.get_updated_frontier_set(inflated_grid, robot, set())

            # Compute the subgoal properties via known_planner
            known_planner.update(
                {'image': step_data['image']},
                vertex_grid,
                new_subgoals,
                step_data['robot_pose'],
                step_data['visibility_mask'])

            # Get the subgoal grid data
            subgoal_grid_data = vertexnav_lsp.learning.get_grid_data_from_obs(
                known_planner.subgoals, rpose, goal,
                max_range=args.max_range,
                num_range=args.num_range,
                num_bearing=args.num_bearing,
                visibility_mask=step_data['visibility_mask'])

            datum.update(subgoal_grid_data)
            write_datum_to_pickle(args, counter, datum)

            if args.do_visualize_data:
                plt.figure()
                plt.subplot(4, 2, 1)
                plt.imshow(datum['image'])
                plt.subplot(4, 2, 3)
                plt.imshow(datum['is_vertex'])
                plt.subplot(4, 2, 5)
                plt.imshow(datum['is_frontier'])
                plt.subplot(4, 2, 7)
                plt.imshow(datum['is_feasible'])
                plt.subplot(2, 2, 2)
                plt.imshow(vertex_grid.T)
                vertexnav.plotting.plot_proposed_world(plt.gca(),
                                                       proposed_world,
                                                       do_show_points=True,
                                                       do_plot_visibility=False,
                                                       robot_pose=v_pose)

                plt.subplot(2, 2, 4)
                plt.imshow(step_data['robot_grid'].T)
                plt.savefig(f"/data/dbg/vertexnav_lsp_{counter:04d}.png", dpi=150)
                plt.show()

            if not do_plan_with_naive:
                planning_loop.set_chosen_subgoal(
                    known_planner.compute_selected_subgoal())

    # Write final plot to file
    image_file = os.path.join(
        args.save_dir, 'data_collect_plots',
        os.path.splitext(args.data_file_base_name)[0] + f'_{args.current_seed}.png')
    print(image_file)

    plt.figure(figsize=(8, 8))
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(known_planner.observed_map, known_map))
    path = robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    p = path[-1]
    plt.plot(ys, xs, 'r')
    plt.plot(p.y, p.x, 'go')
    plt.savefig(image_file, dpi=150)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--xpassthrough', type=str, default='false')
    parser.add_argument('--data_file_base_name', type=str)
    parser.add_argument('--logfile_name', type=str, default='logfile.txt')
    parser.add_argument('--num_range',
                        type=int,
                        help='Number of range cells in output grid.')
    parser.add_argument('--num_bearing',
                        type=int,
                        help='Number of bearing cells in output grid.')
    args = parser.parse_args()
    args.do_visualize_data = args.xpassthrough == 'true'

    args.max_range = args.laser_max_range_m / args.base_resolution

    data_gen_main(args)
