import copy
from lsp.core import get_goal_distances, get_frontier_distances, get_robot_distances, get_top_n_frontiers
import lsp
import gridmap
import numpy as np
import logging
import time
import torch
from .explanation import Explanation
from lsp_xai.learning.models import ExpNavVisLSP

NUM_MAX_FRONTIERS = 12


class SubgoalPlanner(lsp.planners.Planner):
    def __init__(self, goal, args, device=None):
        super(SubgoalPlanner, self).__init__(goal)

        self.subgoals = set()
        self.selected_subgoal = None
        self.observed_map = None
        self.args = args
        self.update_counter = 0

        self.inflation_radius = args.inflation_radius_m / args.base_resolution
        if self.inflation_radius >= np.sqrt(5):
            self.downsample_factor = 2
        else:
            self.downsample_factor = 1

        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device

        self.subgoal_property_net, self.model = ExpNavVisLSP.get_net_eval_fn(
            args.network_file, device=self.device, do_return_model=True)

    def update(self, observation, observed_map, subgoals, robot_pose,
               visibility_mask):
        """Updates the internal state with the new grid/pose/laser scan.

        This function also computes a few necessary items, like the new
        set of frontiers from the inflated grid and their properties from
        the trained network.
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
            max_dist=20.0 / self.args.base_resolution,
            chosen_frontier=self.selected_subgoal)

        # Also check that the goal is not inside the frontier
        lsp.core.update_frontiers_goal_in_frontier(self.subgoals, self.goal)

        self._update_frontier_props_oriented(robot_pose=robot_pose,
                                             goal_pose=self.goal,
                                             image=observation['image'],
                                             visibility_mask=visibility_mask)

    def _update_frontier_props_oriented(self,
                                        robot_pose,
                                        goal_pose,
                                        image,
                                        visibility_mask=None):
        if image is None:
            raise ValueError("argument 'image' must not be 'None'")

        image = image * 1.0 / 255.0

        # Loop through subgoals and set properties
        for subgoal in self.subgoals:
            if subgoal.props_set:
                continue

            # Compute the data that will be passed to the neural net
            input_data = lsp.utils.learning_vision.get_oriented_input_data(
                image, robot_pose, goal_pose, subgoal)

            # Store the input data alongside each subgoal
            subgoal.nn_input_data = input_data

            # Compute subgoal properties from neural network
            [prob_feasible, delta_success_cost, exploration_cost] = \
                self.subgoal_property_net(
                    image=input_data['image'],
                    goal_loc_x=input_data['goal_loc_x'],
                    goal_loc_y=input_data['goal_loc_y'],
                    subgoal_loc_x=input_data['subgoal_loc_x'],
                    subgoal_loc_y=input_data['subgoal_loc_y'])
            subgoal.set_props(prob_feasible=prob_feasible,
                              delta_success_cost=delta_success_cost,
                              exploration_cost=exploration_cost,
                              last_observed_pose=robot_pose)

        for subgoal in self.subgoals:
            if not self.args.silence and subgoal.prob_feasible > 0.0:
                print(" " * 20 + "PLAN  (%.2f %.2f) | %.6f | %7.2f | %7.2f" %
                      (subgoal.get_centroid()[0], subgoal.get_centroid()[1],
                       subgoal.prob_feasible, subgoal.delta_success_cost,
                       subgoal.exploration_cost))

    def _recompute_all_subgoal_properties(self):
        # Loop through subgoals and set properties
        for subgoal in self.subgoals:
            input_data = subgoal.nn_input_data

            # Compute subgoal properties from neural network
            [prob_feasible, delta_success_cost, exploration_cost] = \
                self.subgoal_property_net(
                    image=input_data['image'],
                    goal_loc_x=input_data['goal_loc_x'],
                    goal_loc_y=input_data['goal_loc_y'],
                    subgoal_loc_x=input_data['subgoal_loc_x'],
                    subgoal_loc_y=input_data['subgoal_loc_y'])
            subgoal.set_props(prob_feasible=prob_feasible,
                              delta_success_cost=delta_success_cost,
                              exploration_cost=exploration_cost,
                              last_observed_pose=subgoal.last_observed_pose)

    def compute_selected_subgoal(self):
        is_goal_in_range = lsp.core.goal_in_range(self.inflated_grid,
                                                  self.robot_pose, self.goal,
                                                  self.subgoals)
        if is_goal_in_range:
            print("Goal in Range")
            return None

        # Compute chosen frontier
        min_cost, frontier_ordering = (
            lsp.core.get_best_expected_cost_and_frontier_list(
                self.inflated_grid,
                self.robot_pose,
                self.goal,
                self.subgoals,
                num_frontiers_max=NUM_MAX_FRONTIERS,
                downsample_factor=self.downsample_factor,
                do_correct_low_prob=True))
        if min_cost is None or min_cost > 1e8 or frontier_ordering is None:
            raise ValueError("Problem with planning")

        self.latest_ordering = frontier_ordering
        self.selected_subgoal = frontier_ordering[0]
        return self.selected_subgoal

    def compute_backup_subgoal(self, selected_subgoal):
        subgoals, distances = self.get_subgoals_and_distances()
        return lsp.core.get_lowest_cost_ordering_not_beginning_with(
            selected_subgoal, subgoals, distances)[1][0]

    def compute_subgoal_data(self,
                             chosen_subgoal,
                             num_frontiers_max=NUM_MAX_FRONTIERS,
                             do_return_ind_dict=False):
        is_goal_in_range = lsp.core.goal_in_range(self.inflated_grid,
                                                  self.robot_pose, self.goal,
                                                  self.subgoals)
        if is_goal_in_range:
            return None

        # Compute chosen frontier
        logger = logging.getLogger("SubgoalPlanner")
        stime = time.time()
        policy_data, subgoal_ind_dict = get_policy_data_for_frontiers(
            self.inflated_grid,
            self.robot_pose,
            self.goal,
            chosen_subgoal,
            self.subgoals,
            num_frontiers_max=num_frontiers_max,
            downsample_factor=self.downsample_factor)
        logger.debug(f"time to get policy data: {time.time() - stime}")

        if do_return_ind_dict:
            return policy_data, subgoal_ind_dict
        else:
            return policy_data

    def get_subgoals_and_distances(self, subgoals_of_interest=[]):
        """Helper function for getting data."""
        # Remove frontiers that are infeasible
        subgoals = [s for s in self.subgoals if s.prob_feasible > 0]
        subgoals = list(set(subgoals) | set(subgoals_of_interest))

        # Calculate the distance to the goal and to the robot.
        goal_distances = get_goal_distances(
            self.inflated_grid,
            self.goal,
            frontiers=subgoals,
            downsample_factor=self.downsample_factor)

        robot_distances = get_robot_distances(
            self.inflated_grid,
            self.robot_pose,
            frontiers=subgoals,
            downsample_factor=self.downsample_factor)

        # Get the most n probable frontiers to limit computational load
        if NUM_MAX_FRONTIERS > 0 and NUM_MAX_FRONTIERS < len(subgoals):
            subgoals = get_top_n_frontiers(subgoals, goal_distances,
                                           robot_distances, NUM_MAX_FRONTIERS)
            subgoals = list(set(subgoals) | set(subgoals_of_interest))

        # Calculate robot and frontier distances
        frontier_distances = get_frontier_distances(
            self.inflated_grid,
            frontiers=subgoals,
            downsample_factor=self.downsample_factor)

        distances = {
            'frontier': frontier_distances,
            'robot': robot_distances,
            'goal': goal_distances,
        }

        return subgoals, distances

    def generate_counterfactual_explanation(self,
                                            query_subgoal,
                                            limit_num=-1,
                                            do_freeze_selected=True,
                                            keep_changes=False,
                                            margin=0):
        # Initialize the datum
        device = self.device
        chosen_subgoal = self.compute_selected_subgoal()
        datum, subgoal_ind_dict = self.compute_subgoal_data(
            chosen_subgoal, 24, do_return_ind_dict=True)
        datum = self.model.update_datum(datum, device)

        # Now we want to rearrange things a bit: the new 'target' subgoal we set to
        # our query_subgoal and we populate the 'backup'
        # subgoal with the 'chosen' subgoal (the subgoal the agent actually chose).
        datum['target_subgoal_ind'] = subgoal_ind_dict[query_subgoal]
        if do_freeze_selected:
            datum['backup_subgoal_ind'] = subgoal_ind_dict[chosen_subgoal]

        # We update the datum to reflect this change (and confirm it worked).
        datum = self.model.update_datum(datum, device)
        base_datum = copy.deepcopy(datum)

        # Compute the 'delta subgoal data'. This is how we determine the
        # 'importance' of each of the subgoal properties. In practice, we will sever
        # the gradient for all but a handful of these with an optional parameter
        # (not set here).
        delta_subgoal_data = self.model.get_subgoal_prop_impact(
            datum, device, delta_cost_limit=-1e10)
        base_model_state = self.model.state_dict(keep_vars=False)
        base_model_state = copy.deepcopy(base_model_state)
        base_model_state = {k: v.cpu() for k, v in base_model_state.items()}

        # Initialize some terms for the optimization
        learning_rate = 1.0e-4
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # Now we perfrom iterative gradient descent until the expected cost of the
        # new target subgoal is lower than that of the originally selected subgoal.
        for ii in range(5000):
            # Update datum to reflect new neural network state
            datum = self.model.update_datum(datum, device)

            # Compute the subgoal properties by passing images through the network.
            # (PyTorch implicitly builds a graph of these operations so that we can
            # differentiate them later.)
            nn_out, ind_mapping = self.model(datum, device)
            is_feasibles = torch.nn.Sigmoid()(nn_out[:, 0])
            delta_success_costs = nn_out[:, 1]
            exploration_costs = nn_out[:, 2]
            limited_subgoal_props, _, _ = self.model.compute_subgoal_props(
                is_feasibles,
                delta_success_costs,
                exploration_costs,
                datum['subgoal_data'],
                ind_mapping,
                device,
                limit_subgoals_num=limit_num,
                delta_subgoal_data=delta_subgoal_data)

            # Compute the expected of the new target subgoal:
            q_target = self.model.compute_expected_cost_for_policy(
                limited_subgoal_props, datum['target_subgoal_policy'])
            # Cost of the 'backup' (formerly the agent's chosen subgoal):
            q_backup = self.model.compute_expected_cost_for_policy(
                limited_subgoal_props, datum['backup_subgoal_policy'])
            print(
                f"{ii:5} | Q_dif = {q_target - q_backup:6f} | Q_target = {q_target:6f} | Q_backup = {q_backup:6f}"
            )
            assert q_target > 0
            assert q_backup > 0

            if ii == 0:
                # Store the original values for each.
                base_subgoal_props = limited_subgoal_props

            # The zero-crossing of the difference between the two is the decision
            # boundary we are hoping to cross by updating the paramters of the
            # neural network via gradient descent.
            q_diff = q_target - q_backup

            if q_diff <= -margin:
                # When it's less than zero, we're done.
                upd_subgoal_props = limited_subgoal_props
                break

            # Via PyTorch magic, gradient descent is easy:
            optimizer.zero_grad()
            q_diff.backward()
            optimizer.step()
        else:
            # If it never crossed the boundary, we have failed.
            raise ValueError("Decision boundary never crossed.")

        # Restore the model to its previous value
        if not keep_changes:
            print("Restoring Model")
            self.model.load_state_dict(base_model_state)
            self.model.eval()
            self.model = self.model.to(device)
        else:
            print("Keeping model")
            self._recompute_all_subgoal_properties()

        upd_subgoal_props = limited_subgoal_props
        return Explanation(self.subgoals, subgoal_ind_dict, base_datum,
                           base_subgoal_props, datum, upd_subgoal_props,
                           delta_subgoal_data, self.observed_map,
                           self.inflated_grid, self.goal, self.robot_pose,
                           limit_num)


# Alt versions of functions
def get_policy_data_for_frontiers(grid,
                                  robot_pose,
                                  goal_pose,
                                  chosen_frontier,
                                  all_frontiers,
                                  num_frontiers_max=0,
                                  downsample_factor=1):
    """Compute the optimal orderings for each frontier of interest and return a data
structure containing all the information that would be necessary to compute the
expected cost for each. Also returns the mapping from 'frontiers' to 'inds'."""

    # Remove frontiers that are infeasible
    frontiers = [f for f in all_frontiers if f.prob_feasible != 0]
    frontiers = list(set(frontiers) | set([chosen_frontier]))

    # Calculate the distance to the goal, if infeasible, remove frontier
    goal_distances = get_goal_distances(grid,
                                        goal_pose,
                                        frontiers=frontiers,
                                        downsample_factor=downsample_factor)

    robot_distances = get_robot_distances(grid,
                                          robot_pose,
                                          frontiers=frontiers,
                                          downsample_factor=downsample_factor)

    # Get the most n probable frontiers to limit computational load
    if num_frontiers_max > 0 and num_frontiers_max < len(frontiers):
        frontiers = lsp.core.get_top_n_frontiers_distance(
            frontiers, goal_distances, robot_distances, num_frontiers_max)
        frontiers = list(set(frontiers) | set([chosen_frontier]))

    # Calculate robot and frontier distances
    frontier_distances = get_frontier_distances(
        grid, frontiers=frontiers, downsample_factor=downsample_factor)

    frontier_ind_dict = {f: ind for ind, f in enumerate(frontiers)}
    robot_distances_ind = {
        frontier_ind_dict[f]: robot_distances[f]
        for f in frontiers
    }
    goal_distances_ind = {
        frontier_ind_dict[f]: goal_distances[f]
        for f in frontiers
    }
    frontier_distances_ind = {}
    for ind, f1 in enumerate(frontiers[:-1]):
        f1_ind = frontier_ind_dict[f1]
        for f2 in frontiers[ind + 1:]:
            f2_ind = frontier_ind_dict[f2]
            frontier_distances_ind[frozenset(
                [f1_ind, f2_ind])] = (frontier_distances[frozenset([f1, f2])])

    if frontier_distances is not None:
        assert len(frontier_distances.keys()) == len(
            frontier_distances_ind.keys())

    # Finally, store the data relevant for
    # estimating the frontier properties
    frontier_data = {
        ind: f.nn_input_data
        for f, ind in frontier_ind_dict.items()
    }

    return {
        'subgoal_data': frontier_data,
        'distances': {
            'frontier': frontier_distances_ind,
            'robot': robot_distances_ind,
            'goal': goal_distances_ind,
        },
        'target_subgoal_ind': frontier_ind_dict[chosen_frontier]
    }, frontier_ind_dict
