import os
import environments
import numpy as np
import matplotlib.pyplot as plt
import lsp
from lsp_cond.planners import ConditionalKnownSubgoalPlanner
import gridmap
import lsp_cond
import torch
from matplotlib import cm
import matplotlib.animation as animation


viridis = cm.get_cmap('viridis')


def navigate(args, do_plot=False, make_video=False):
    if make_video:
        fig = plt.figure()
        writer = animation.FFMpegWriter(15)
        writer.setup(fig, os.path.join(args.save_dir, 
                     f'{args.data_file_base_name}_{args.current_seed}.mp4'), 500)
    known_map, map_data, pose, goal = \
        environments.generate.map_and_poses(args)
    # Change the initial robot pose
    if args.do_randomize_start_pose:
        if not args.map_type == 'maze':
            raise ValueError('Can only randomize/change start pose for Maze environment.')
        else:
            pose = lsp_cond.utils.get_path_middle_point(known_map, pose, goal, args)
    # Instantiate the simulation environment
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements)
    builder = environments.simulated.WorldBuildingUnityBridge

    robot = lsp.robot.Turtlebot_Robot(pose,
                                      primitive_length=args.step_size,
                                      num_primitives=args.num_primitives,
                                      map_data=map_data)

    # Intialize and update the planner
    planner = ConditionalKnownSubgoalPlanner(goal, args, known_map)

    with builder(args.unity_path) as unity_bridge:
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
            # Update the planner objects
            planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'])
            training_data = planner.compute_training_data()
            planner.save_training_data(training_data)
            chosen_subgoal = planner.compute_selected_subgoal()
            planning_loop.set_chosen_subgoal(chosen_subgoal)
            
            if do_plot:
                # Mask grid with chosen subgoal (if not None)
                # and compute the cost grid for motion planning.
                if chosen_subgoal is not None:
                    planning_grid = lsp.core.mask_grid_with_frontiers(
                        planner.inflated_grid, planner.subgoals, do_not_mask=chosen_subgoal)
                else:
                    planning_grid = lsp.core.mask_grid_with_frontiers(
                        planner.inflated_grid,
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
                for (s, e) in planner.edge_data:
                    ps = planner.current_graph[s][e]['pts']
                    plt.plot(ps[:, 0], ps[:, 1], 'lightgray')
                
                is_subgoal = training_data['is_subgoal']
                for vp_idx, _ in enumerate(planner.vertex_points):
                    ps = planner.vertex_points[vp_idx]
                    color = viridis(1.0 * is_subgoal[vp_idx])
                    plt.plot(ps[0], ps[1], '.', color=color, markersize=3)
                plt.show()
                image_file = '/data/lsp_conditional/test_image' + '.png'
                plt.savefig(image_file, dpi=150)
                if make_video:
                    writer.grab_frame()
        
        open(os.path.join(
            args.save_dir, 
            'data_completion_logs',
            f'{args.data_file_base_name}_{args.current_seed}.txt'), "x")
        if make_video:
            writer.finish()


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    args = lsp_cond.utils.parse_args()
    # Always freeze your random seeds
    torch.manual_seed(8616)
    np.random.seed(8616)
    # # Generate Training Data
    print(args.current_seed)
    navigate(args, make_video=False, do_plot=False)
