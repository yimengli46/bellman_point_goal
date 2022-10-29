import os
import torch
import copy
from lsp.planners.planner import Planner
import lsp
import gridmap
import lsp_cond
import numpy as np
from lsp_cond.learning.models.auto_encoder import AutoEncoder
from lsp_cond.learning.models.gcn import LSPConditionalGNN\


NUM_MAX_FRONTIERS = 12


class ConditionalSubgoalPlanner(Planner):
    def __init__(self, goal, args, device=None):
        super(ConditionalSubgoalPlanner, self).__init__(goal)

        self.subgoals = set()
        self.selected_subgoal = None
        self.args = args

        self.vertex_points = None
        self.edge_data = None

        self.inflation_radius = args.inflation_radius_m / args.base_resolution
        if self.inflation_radius >= np.sqrt(5):
            self.downsample_factor = 2
        else:
            self.downsample_factor = 1

        self.old_node_dict = {}

    def update(self, observation, observed_map, subgoals, robot_pose):
        """Updates the internal state with the new grid/pose/laser scan.
        This function also computes a few necessary items, like which
        frontiers have recently been updated and computes their properties
        from the known grid.
        """
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
            max_dist=2.0 / self.args.base_resolution,  # Was set to 20.0
            chosen_frontier=self.selected_subgoal)

        # Also check that the goal is not inside the frontier
        lsp.core.update_frontiers_goal_in_frontier(self.subgoals,
                                                   # robot_pose, 
                                                   self.goal)

        vertex_data, self.edge_data, self.current_graph = lsp_cond.utils. \
            compute_skeleton(inflated_grid.copy(), self.subgoals)
            
        self.vertex_points = np.array([vertex_data[i]['o'] 
                                      for i in vertex_data])

        # Update the vertex point inputs
        self._update_node_inputs(self.vertex_points)
                
        # Update the subgoal inputs & get representative nodes for the subgoals in the graph
        self.subgoal_nodes = self._identify_subgoal_nodes(
            self.subgoals.copy(), 
            self.vertex_points)

        # Once the subgoal inputs are set, compute their properties
        self._update_subgoal_properties(robot_pose, self.goal)
        self.old_node_dict = self.new_node_dict.copy()

    def _identify_subgoal_nodes(self, subgoals, vertex_points):
        # Loop through subgoals and get the 'input data'
        """ This method also finds the representitive node for each subgoal
        on the graph and pairs their image as well
        """
        subgoal_nodes = {}
        for subgoal in subgoals:
            possible_node = tuple(
                lsp_cond.utils.get_subgoal_node(vertex_points, subgoal))
            if possible_node in subgoal_nodes.keys():
                # raise ValueError("Duplicate Association <- Need to remove duplicate subgoal(s)")
                open(os.path.join(
                    '/data/lsp_conditional/error_logs/',
                    f'Duplication_on_{self.args.current_seed}.txt'), "x")
            else:
                subgoal_nodes[possible_node] = subgoal
        # # The line below removes the duplicate frotier from the set
        self.subgoals = set(list(subgoal_nodes.values()))
        return subgoal_nodes
    
    def _update_node_inputs(self, vertex_points):
        ''' This method computes and assigns input for each of the nodes
        present on the graph and maintains a dictionary for it.
        '''
        self.new_node_dict = {}
        for vertex_point in vertex_points:
            vertex_point = tuple(vertex_point)
            # If the vertex point exists in previous step then perform the
            # following steps ->
            if vertex_point in self.old_node_dict.keys():
                # If new robot_pose is closer than the last observed pose,
                # then update the inputs
                if lsp_cond.utils.is_closer(
                    self.robot_pose,
                    self.old_node_dict[vertex_point]['last_observed_pose'],
                    vertex_point, self.inflated_grid
                ):
                    input_data = lsp_cond.utils. \
                        get_oriented_non_subgoal_input_data(
                            self.observation['image'],
                            self.robot_pose,
                            self.goal, vertex_point)
                    input_data['last_observed_pose'] = self.robot_pose
                else:
                    input_data = self.old_node_dict[vertex_point]
            # -> Otherwise calculate input data for new vertex point
            else:
                input_data = lsp_cond.utils. \
                    get_oriented_non_subgoal_input_data(
                        self.observation['image'],
                        self.robot_pose,
                        self.goal, vertex_point)
                input_data['last_observed_pose'] = self.robot_pose
            self.new_node_dict[vertex_point] = input_data

    def _compute_combined_data(self):
        """ This method produces a datum for the GCN and returns it.
        make_graph(datum) needs to be called prior to forwording to 
        the network
        """
        is_subgoal = []
        history = []  
        images = []
        goal_loc_x = []
        goal_loc_y = []
        subgoal_loc_x = []
        subgoal_loc_y = []
     
        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            images.append(self.new_node_dict[vertex_point]['image'])
            goal_loc_x.append(self.new_node_dict[vertex_point]['goal_loc_x'])
            goal_loc_y.append(self.new_node_dict[vertex_point]['goal_loc_y'])
            subgoal_loc_x.append(self.new_node_dict[vertex_point]['subgoal_loc_x'])
            subgoal_loc_y.append(self.new_node_dict[vertex_point]['subgoal_loc_y'])
            if vertex_point in self.subgoal_nodes.keys():
                # taking note of the node being a subgoal
                is_subgoal.append(1)
                # marking if the subgoal will participate 
                # in conditional probability
                history.append(1)                 
            else:
                is_subgoal.append(0)
                history.append(0)
        
        datum = {
            'image': images,
            'goal_loc_x': goal_loc_x,
            'goal_loc_y': goal_loc_y,
            'subgoal_loc_x': subgoal_loc_x,
            'subgoal_loc_y': subgoal_loc_y,
            'is_subgoal': is_subgoal,
            'history': history,
            'edge_data': self.edge_data
        }
        return datum

    def _update_subgoal_properties(self, robot_pose, goal_pose):
        raise NotImplementedError("Method for abstract class")


class ConditionalKnownSubgoalPlanner(ConditionalSubgoalPlanner):
    def __init__(self, goal, args, known_map, device=None, do_compute_weightings=True):
        super(ConditionalKnownSubgoalPlanner, self). \
            __init__(goal, args, device)

        self.known_map = known_map
        self.inflated_known_grid = gridmap.utils.inflate_grid(
            known_map, inflation_radius=self.inflation_radius)
        _, self.get_path = gridmap.planning.compute_cost_grid_from_position(
            self.inflated_known_grid, [goal.x, goal.y])
        self.counter = 0
        self.do_compute_weightings = do_compute_weightings

    def _update_subgoal_properties(self, robot_pose, goal_pose):
        new_subgoals = [s for s in self.subgoals if not s.props_set]
        lsp.core.update_frontiers_properties_known(
            self.inflated_known_grid,
            self.inflated_grid,
            self.subgoals, new_subgoals,
            robot_pose, goal_pose,
            self.downsample_factor)

        if self.do_compute_weightings:
            lsp.core.update_frontiers_weights_known(self.inflated_known_grid,
                                                    self.inflated_grid,
                                                    self.subgoals, new_subgoals,
                                                    robot_pose, goal_pose,
                                                    self.downsample_factor)

    def compute_training_data(self):
        """ This method produces training datum for both AutoEncoder and
        GCN model.
        """
        prob_feasible = []
        delta_success_cost = []
        exploration_cost = []
        positive_weighting_vector = []
        negative_weighting_vector = []

        for node in self.vertex_points:
            p = tuple(node)
            if p in self.subgoal_nodes.keys():
                prob_feasible.append(self.subgoal_nodes[p].prob_feasible)
                delta_success_cost.append(self.subgoal_nodes[p].delta_success_cost)
                exploration_cost.append(self.subgoal_nodes[p].exploration_cost)
                positive_weighting_vector.append(self.subgoal_nodes[p].positive_weighting)
                negative_weighting_vector.append(self.subgoal_nodes[p].negative_weighting)
            else:
                prob_feasible.append(0)
                delta_success_cost.append(0)
                exploration_cost.append(0)
                positive_weighting_vector.append(0)
                negative_weighting_vector.append(0)

        data = self._compute_combined_data()
        data['is_feasible'] = prob_feasible
        data['delta_success_cost'] = delta_success_cost
        data['exploration_cost'] = exploration_cost
        data['positive_weighting'] = positive_weighting_vector
        data['negative_weighting'] = negative_weighting_vector
        
        return data

    def save_training_data(self, training_data):
        lsp_cond.utils.write_datum_to_file(self.args,
                                           training_data, 
                                           self.counter)
        self.counter += 1

    def compute_selected_subgoal(self):
        """Use the known map to compute the selected subgoal."""

        if not self.subgoals:
            return None

        # Compute the plan
        did_plan, path = self.get_path([self.robot_pose.x, self.robot_pose.y],
                                       do_sparsify=False,
                                       do_flip=True,
                                       bound=None)
        if did_plan is False:
            print("Plan did not succeed...")
            raise NotImplementedError("Not sure what to do here yet")
        if np.argmax(self.observed_map[path[0, -1], path[1, -1]] >= 0):
            return None

        # Determine the chosen subgoal
        ind = np.argmax(self.observed_map[path[0, :], path[1, :]] < 0)
        return min(self.subgoals,
                   key=lambda s: s.get_distance_to_point((path.T)[ind]))


class ConditionalUnknownSubgoalPlanner(ConditionalSubgoalPlanner):
    def __init__(self, goal, args, device=None):
        super(ConditionalUnknownSubgoalPlanner, self). \
            __init__(goal, args, device)

        self.out = None       

        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device

        self.latent_features_net = AutoEncoder. \
            get_net_eval_fn(args.autoencoder_network_file, device=self.device)
        self.subgoal_property_net = LSPConditionalGNN. \
            get_net_eval_fn(args.network_file, device=self.device)

    def _compute_cnn_data(self):
        """ This method produces datum for the AutoEncoder """
        images = []
        goal_loc_x = []
        goal_loc_y = []
        subgoal_loc_x = []
        subgoal_loc_y = []
        
        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            # Check if the latent features need to be recomputed for the vertex
            # points
            if 'latent_features' not in self.new_node_dict[vertex_point].keys():
                images.append(self.new_node_dict[vertex_point]['image'])
                goal_loc_x.append(self.new_node_dict[vertex_point]['goal_loc_x'])
                goal_loc_y.append(self.new_node_dict[vertex_point]['goal_loc_y'])
                subgoal_loc_x.append(self.new_node_dict[vertex_point]['subgoal_loc_x'])
                subgoal_loc_y.append(self.new_node_dict[vertex_point]['subgoal_loc_y'])
      
        datum = {
            'image': images,
            'goal_loc_x': goal_loc_x,
            'goal_loc_y': goal_loc_y,
            'subgoal_loc_x': subgoal_loc_x,
            'subgoal_loc_y': subgoal_loc_y,
        }
        return datum

    def _calculate_latent_features(self):
        ''' This method computes and assigns latent features to their 
        respective node.
        '''
        self.cnn_input = self._compute_cnn_data()
        if self.cnn_input['image']:  # Check if cnn_input is not empty
            latent_features = self.latent_features_net(
                datum=self.cnn_input)
            _, length = latent_features.shape
        ii = 0
        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            # Checks and assigns latent features to their nodes
            if 'latent_features' not in self.new_node_dict[vertex_point].keys():
                self.new_node_dict[vertex_point]['latent_features'] = \
                    latent_features[ii].view(1, length)
                ii += 1

    def _compute_gcn_data(self):
        """ This method produces a datum for the GCN and returns it.
        make_graph(datum) needs to be called prior to forwording to 
        the network
        """
        # Prior to running GCN, CNN must create the latent features
        self._calculate_latent_features()
        latent_features = torch.zeros(0).to(self.device)
        is_subgoal = []
        history = []
     
        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            latent_features = torch.cat(
                (latent_features, 
                    self.new_node_dict[vertex_point]['latent_features']), 0)
            if vertex_point in self.subgoal_nodes.keys():
                # taking note of the node being a subgoal
                is_subgoal.append(1)
                # marking if the subgoal will participate 
                # in conditional probability
                history.append(1)
            else:
                is_subgoal.append(0)
                history.append(0)
      
        datum = {
            'is_subgoal': is_subgoal,
            'history': history,
            'edge_data': self.edge_data,
            'latent_features': latent_features
        }
        return datum
    
    def _update_subgoal_properties(self,
                                   robot_pose,
                                   goal_pose):
        self.gcn_graph_input = self._compute_gcn_data()
        prob_feasible_dict, dsc, ec, out = self.subgoal_property_net(
            datum=self.gcn_graph_input,
            vertex_points=self.vertex_points,
            subgoals=self.subgoals
        )
        for subgoal in self.subgoals:
            subgoal.set_props(
                prob_feasible=prob_feasible_dict[subgoal],
                delta_success_cost=dsc[subgoal],
                exploration_cost=ec[subgoal],
                last_observed_pose=robot_pose)
        self.out = out
        # The line below is not getting updated to the last elimination pass
        # that is happening before cost calculation, however this is not that
        # important because the self.is_subgoal is only used to plot the graph
        # not in cost calculation. Will fix it if necessary.
        self.is_subgoal = self.gcn_graph_input['is_subgoal'].copy()               
    
    def compute_selected_subgoal(self):
        is_goal_in_range = lsp.core.goal_in_range(self.inflated_grid,
                                                  self.robot_pose,
                                                  self.goal, self.subgoals)
        if is_goal_in_range:
            print("Goal in Range")
            return None
        # Compute chosen frontier
        min_cost, frontier_ordering = (
            lsp_cond.core.get_best_expected_cost_and_frontier_list(
                self.inflated_grid,
                self.robot_pose,
                self.goal,
                self.subgoals,
                self.vertex_points.copy(),
                self.subgoal_nodes.copy(),
                self.gcn_graph_input.copy(),
                self.subgoal_property_net,
                num_frontiers_max=NUM_MAX_FRONTIERS,
                downsample_factor=self.downsample_factor))
        if min_cost is None or min_cost > 1e8 or frontier_ordering is None:
            raise ValueError()
            print("Problem with planning.")
            return None
        self.latest_ordering = frontier_ordering
        self.selected_subgoal = list(self.subgoals)[frontier_ordering[0]]
        return self.selected_subgoal


class ConditionalCombinedPlanner(ConditionalSubgoalPlanner):
    def __init__(self, goal, args, device=None):
        super(ConditionalCombinedPlanner, self). \
            __init__(goal, args, device)
        self.out = None       
        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        self.latent_features_net = AutoEncoder. \
            get_net_eval_fn(args.autoencoder_network_file, device=self.device)
        self.subgoal_property_net = LSPConditionalGNN. \
            get_net_eval_fn(args.network_file, device=self.device)
        self.counter = 0

    def _compute_cnn_data(self):
        """ This method produces datum for the AutoEncoder """
        images = []
        goal_loc_x = []
        goal_loc_y = []
        subgoal_loc_x = []
        subgoal_loc_y = []
        
        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            images.append(self.new_node_dict[vertex_point]['image'])
            goal_loc_x.append(self.new_node_dict[vertex_point]['goal_loc_x'])
            goal_loc_y.append(self.new_node_dict[vertex_point]['goal_loc_y'])
            subgoal_loc_x.append(self.new_node_dict[vertex_point]['subgoal_loc_x'])
            subgoal_loc_y.append(self.new_node_dict[vertex_point]['subgoal_loc_y'])
      
        datum = {
            'image': images,
            'goal_loc_x': goal_loc_x,
            'goal_loc_y': goal_loc_y,
            'subgoal_loc_x': subgoal_loc_x,
            'subgoal_loc_y': subgoal_loc_y,
        }
        return datum

    def _compute_gcn_data(self, latent_features):
        """ This method produces a datum for the GCN and returns it.
        make_graph(datum) needs to be called prior to forwording to 
        the network
        """
        is_subgoal = []
        history = []
     
        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            if vertex_point in self.subgoal_nodes.keys():
                # taking note of the node being a subgoal
                is_subgoal.append(1)
                # marking if the subgoal will participate 
                # in conditional probability
                history.append(1)
            else:
                is_subgoal.append(0)
                history.append(0)
      
        datum = {
            'is_subgoal': is_subgoal,
            'history': history,
            'edge_data': self.edge_data
        }
        datum['latent_features'] = latent_features
        return datum

    def _calculate_latent_features(self):
        self.cnn_input = self._compute_cnn_data()
        latent_features = self.latent_features_net(
            datum=self.cnn_input)
        return latent_features
    
    def _update_subgoal_properties(self,
                                   robot_pose,
                                   goal_pose):
        latent_features = self._calculate_latent_features()
        self.lf = latent_features
        self.gcn_graph_input = self._compute_gcn_data(latent_features)
        prob_feasible_dict, dsc, ec, out = self.subgoal_property_net(
            datum=self.gcn_graph_input,
            vertex_points=self.vertex_points,
            subgoals=self.subgoals
        )
        for subgoal in self.subgoals:
            subgoal.set_props(
                prob_feasible=prob_feasible_dict[subgoal],
                delta_success_cost=dsc[subgoal],
                exploration_cost=ec[subgoal],
                last_observed_pose=robot_pose)
        self.out = out
        # The line below is not getting updated to the last elimination pass
        # that is happening before cost calculation, however this is not that
        # important because the self.is_subgoal is only used to plot the graph
        # not in cost calculation. Will fix it if necessary.
        self.is_subgoal = self.gcn_graph_input['is_subgoal'].copy()               
    
    def compute_selected_subgoal(self):
        is_goal_in_range = lsp.core.goal_in_range(self.inflated_grid,
                                                  self.robot_pose,
                                                  self.goal, self.subgoals)
        if is_goal_in_range:
            print("Goal in Range")
            return None
        # Compute chosen frontier
        min_cost, frontier_ordering = (
            lsp_cond.core.get_best_expected_cost_and_frontier_list(
                self.inflated_grid,
                self.robot_pose,
                self.goal,
                self.subgoals,
                self.vertex_points.copy(),
                self.subgoal_nodes.copy(),
                self.gcn_graph_input.copy(),
                self.subgoal_property_net,
                num_frontiers_max=NUM_MAX_FRONTIERS,
                downsample_factor=self.downsample_factor))
        if min_cost is None or min_cost > 1e8 or frontier_ordering is None:
            raise ValueError()
            print("Problem with planning.")
            return None
        self.latest_ordering = frontier_ordering
        self.selected_subgoal = list(self.subgoals)[frontier_ordering[0]]
        return self.selected_subgoal
