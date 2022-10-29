import copy
import gridmap
import lsp
from lsp.planners import LearnedSubgoalPlanner
import shapely.geometry
import vertexnav


class LearnedVertexnavSubgoalPlanner(LearnedSubgoalPlanner):
    def __init__(self, goal, args, verbose=False):
        super(LearnedSubgoalPlanner, self).__init__(goal, args)

    def update(self, observation, observed_map, subgoals, robot_pose,
               visibility_polygon):
        """Updates the internal state with the new grid/pose/laser scan.

        This function also computes a few necessary items, like which
        frontiers have recently been updated and computes their properties
        from the known grid.
        """
        self.update_counter += 1
        self.observation = observation
        self.observed_map = observed_map
        self.robot_pose = robot_pose

        # Store the inflated grid after ensuring that the unreachable 'free
        # space' is set to 'unobserved'. This avoids trying to plan to
        # unreachable space and avoids having to check for this everywhere.
        inflated_grid = self._get_inflated_occupancy_grid()
        self.inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
            inflated_grid, robot_pose)

        # Compute the new frontiers and update stored frontiers
        new_subgoals = set([copy.copy(s) for s in subgoals])
        self.subgoals = lsp.core.update_frontier_set(
            self.subgoals,
            new_subgoals,
            max_dist=2.0 / self.args.base_resolution,
            chosen_frontier=self.selected_subgoal)

        # Also check that the goal is not inside the frontier
        lsp.core.update_frontiers_goal_in_frontier(self.subgoals, self.goal)

        # Update the subgoal properties using the learned data from the observation
        # Store some helper vars
        max_range = self.args.laser_max_range_m / self.args.base_resolution
        num_range = self.args.num_range
        num_bearing = self.args.num_bearing
        robot_pose = vertexnav.Pose(robot_pose.x, robot_pose.y, robot_pose.yaw)

        # Lookup vectors
        vec_range, vec_bearing = lsp.utils.learning_vision.get_range_bearing_vecs(
            max_range, num_range, num_bearing)

        prob_feasible_mat = observation['subgoal_prob_feasible']
        delta_success_cost_mat = observation['subgoal_delta_success_cost']
        exploration_cost_mat = observation['subgoal_exploration_cost']

        # Compute the frontier properties
        for subgoal in self.subgoals:
            if subgoal.props_set:
                continue

            # Get the range-bearing indices
            prob_feasible_s = 0.0
            delta_success_cost = 0.0
            exploration_cost = 0.0
            counter = 0
            for p in subgoal.points.T:
                is_inside, ind_range, ind_bearing = lsp.utils.learning_vision.get_range_bearing_indices(
                    obs_pose=robot_pose, lookup_point=p,
                    vec_bearing=vec_bearing, vec_range=vec_range
                )
                if visibility_polygon is not None:
                    is_inside = (
                        is_inside and
                        visibility_polygon.contains(shapely.geometry.Point(
                            p[0] / self.args.base_resolution, p[1] / self.args.base_resolution))
                    )

                if is_inside:
                    prob_feasible_s += prob_feasible_mat[ind_range, ind_bearing]
                    delta_success_cost += delta_success_cost_mat[ind_range, ind_bearing]
                    exploration_cost += exploration_cost_mat[ind_range, ind_bearing]
                    counter += 1

            if counter > 0:
                subgoal.set_props(
                    prob_feasible=prob_feasible_s / counter,
                    delta_success_cost=delta_success_cost / counter,
                    exploration_cost=exploration_cost / counter,
                )
            else:
                subgoal.is_set = True

            if self.verbose:
                print(" " * 20 + "PLAN  (%.2f %.2f) | %.6f | %7.2f | %7.2f" % (
                    subgoal.get_centroid()[0],
                    subgoal.get_centroid()[1],
                    subgoal.prob_feasible,
                    subgoal.delta_success_cost,
                    subgoal.exploration_cost))
