import itertools
import lsp
import gridmap
import common
import math
import torch
import os
import learning
import glob
import sknw
import numpy as np
import random
from skimage.morphology import skeletonize
from torch_geometric.data import Data
from lsp.constants import UNOBSERVED_VAL

COMPRESS_LEVEL = 2


def get_n_frontier_points_nearest_to_centroid(subgoal, n):
    """ Get n points from the sorted frontier points' middle indices
    """
    total_points_on_subgoal = len(subgoal.points[0])
    if total_points_on_subgoal <= n:
        return subgoal.points
    mid = total_points_on_subgoal // 2
    return subgoal.points[:, mid - 1:mid + 1]
    

def compute_skeleton(partial_map, subgoals):
    """Perfom skeletonization on free+unknown space image
    """
    n = 2  # Set using trial and error
    free_unknown_image = \
        lsp.core.mask_grid_with_frontiers(partial_map, subgoals)
    for subgoal in subgoals:
        points_to_be_opened = \
            get_n_frontier_points_nearest_to_centroid(subgoal, n)
        for idx, _ in enumerate(points_to_be_opened[0]):
            x = points_to_be_opened[0][idx]
            y = points_to_be_opened[1][idx]
            free_unknown_image[x][y] = 0
            if partial_map[x][y] == UNOBSERVED_VAL:
                partial_map[x][y] = 0

    free_unknown_image = free_unknown_image != 1
    sk = skeletonize(free_unknown_image)
    sk[partial_map == UNOBSERVED_VAL] = 0

    graph = sknw.build_sknw(sk)

    return graph.nodes(), graph.edges(), graph


def calculate_euclidian_distance(node, subgoal):
    return ((node[0] - subgoal[0])**2 + (node[1] - subgoal[1])**2)**.5


def get_subgoal_node(vertex_points, subgoal):
    possible_node = 0
    distance = 10000
    subgoal_centroid = subgoal.get_centroid()
    for node in vertex_points:
        d = calculate_euclidian_distance(node, subgoal_centroid)
        if d < distance:
            distance = d
            possible_node = node
    return possible_node


def is_closer(new_pose, old_pose, node, inflated_grid):
    """ Checks and returns true if the new_pose is closer to the graph
    node than the old_pose
    """
    _, get_path = gridmap.planning.compute_cost_grid_from_position(
        inflated_grid,
        [new_pose.x, new_pose.y],
        use_soft_cost=True
    )
    did_plan, path = get_path(
        node,
        do_sparsify=False,
        do_flip=True,
        bound=None)
    new_distance = common.compute_path_length(path)
    _, get_path = gridmap.planning.compute_cost_grid_from_position(
        inflated_grid,
        [old_pose.x, old_pose.y],
        use_soft_cost=True
    )
    did_plan, path = get_path(
        node,
        do_sparsify=False,
        do_flip=True,
        bound=None)
    old_distance = common.compute_path_length(path)
    if new_distance < old_distance:
        return True
    return False


def preprocess_cnn_data(datum):
    data = datum.copy()
    data['image'] = torch.tensor(np.transpose(data['image'], 
                                 (0, 3, 1, 2)).astype(np.float32) / 255).float()
    data['goal_loc_x'] = torch.tensor(data['goal_loc_x']).float()
    data['goal_loc_y'] = torch.tensor(data['goal_loc_y']).float()
    data['subgoal_loc_x'] = torch.tensor(data['subgoal_loc_x']).float()
    data['subgoal_loc_y'] = torch.tensor(data['subgoal_loc_y']).float() 
    return data

# Fixe me: The function below and above should be merged


def preprocess_autoencoder_data(datum):
    data = {}
    data['image'] = torch.tensor((np.transpose(datum['image'][0],
                                 (2, 0, 1)).astype(np.float32) / 255), 
                                 dtype=torch.float)
    data['goal_loc_x'] = torch.tensor(datum['goal_loc_x'][0], dtype=torch.float)
    data['goal_loc_y'] = torch.tensor(datum['goal_loc_y'][0], dtype=torch.float)
    data['subgoal_loc_x'] = torch.tensor(datum['subgoal_loc_x'][0], dtype=torch.float)
    data['subgoal_loc_y'] = torch.tensor(datum['subgoal_loc_y'][0], dtype=torch.float)
    return data


def preprocess_encoder_batch(batch):
    datum = {
        'image': batch.x,
        'goal_loc_x': batch.glx,
        'goal_loc_y': batch.gly,
        'subgoal_loc_x': batch.slx,
        'subgoal_loc_y': batch.sly,
    }
    return datum


def preprocess_gcn_data(datum):
    data = datum.copy()
    temp = [[x[0], x[1]] for x in data['edge_data']]
    data['edge_data'] = torch.tensor(list(zip(*temp)), dtype=torch.long)
    data['history'] = torch.tensor(data['history'], dtype=torch.long)
    data['is_subgoal'] = torch.tensor(data['is_subgoal'], 
                                      dtype=torch.long)
    return data


def preprocess_gcn_training_data(flag):  # flag = marginal/random
    ''' This method preprocesses the data for GCN training with two options
    marginal history and random history
    '''
    def make_graph(datum):  
        data = datum.copy()
        data['image'] = torch.tensor((np.transpose(data['image'],
                                     (0, 3, 1, 2)).astype(np.float32) / 255), 
                                     dtype=torch.float)          
        
        temp = [[x[0], x[1]] for x in data['edge_data']]
        data['edge_data'] = torch.tensor(list(zip(*temp)), dtype=torch.long)
        
        history = data['history'].copy()
        data['history'] = torch.tensor(data['history'], dtype=torch.long)
        data['is_subgoal'] = torch.tensor(data['is_subgoal'], dtype=torch.long)
        data['goal_loc_x'] = torch.tensor(data['goal_loc_x'], dtype=torch.float)
        data['goal_loc_y'] = torch.tensor(data['goal_loc_y'], dtype=torch.float)
        data['subgoal_loc_x'] = torch.tensor(data['subgoal_loc_x'], dtype=torch.float)
        data['subgoal_loc_y'] = torch.tensor(data['subgoal_loc_y'], dtype=torch.float)
        
        label = data['is_feasible'].copy()
        data['is_feasible'] = torch.tensor(data['is_feasible'], dtype=torch.float)
        data['delta_success_cost'] = torch.tensor(
            data['delta_success_cost'], dtype=torch.float)
        data['exploration_cost'] = torch.tensor(
            data['exploration_cost'], dtype=torch.float)
        data['positive_weighting'] = torch.tensor(
            data['positive_weighting'], dtype=torch.float)
        data['negative_weighting'] = torch.tensor(
            data['negative_weighting'], dtype=torch.float)
        tg_GCN_format = Data(x=data['image'],
                             edge_index=data['edge_data'],
                             is_subgoal=data['is_subgoal'],
                             glx=data['goal_loc_x'],
                             gly=data['goal_loc_y'],
                             slx=data['subgoal_loc_x'],
                             sly=data['subgoal_loc_y'],
                             y=data['is_feasible'],
                             dsc=data['delta_success_cost'],
                             ec=data['exploration_cost'],
                             pweight=data['positive_weighting'],
                             nweight=data['negative_weighting'])

        if flag == 'marginal':
            # Formating for training only with marginal history vector
            tg_GCN_format.__setitem__('history', data['history'])
        elif flag == 'random':
            # Formating for training with randomly chosen 
            # history vector
            history_vectors = generate_all_history_combination(
                history, label)
            history_vector = random.choice(history_vectors)
            data['history'] = torch.tensor(history_vector, 
                                           dtype=torch.long)
            tg_GCN_format.__setitem__('history', data['history'])
        result = tg_GCN_format
        return result
    return make_graph


def generate_all_history_combination(history, node_labels):
    """ Generate all possible combination of subgoals including the one 
    that lead to the goal
    """
    pool = [i for i, val in enumerate(history) 
            if val == 1 and node_labels[i] == 0]
    n = history.count(1)
    c = node_labels.count(1)
    history_vectors = []
    for idx in range(n - c + 1):
        combis = itertools.combinations(pool, idx)
        for a_tuple in combis:
            temp = history.copy()
            for val in a_tuple:
                temp[val] = 0
            history_vectors.append(temp)
    return history_vectors


def generate_all_rollout_history(history):
    pool = [i for i, val in enumerate(history) 
            if val == 1]
    n = history.count(1)
    history_vectors = []
    for idx in range(n):
        combis = itertools.combinations(pool, idx)
        for a_tuple in combis:
            temp = history.copy()
            for val in a_tuple:
                temp[val] = 0
            history_vectors.append(temp)
    return history_vectors


def write_datum_to_file(args, datum, counter):
    """Write a single datum to file and append name to csv record."""
    # Get the data file name
    data_filename = os.path.join('pickles', f'dat_{args.current_seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(
        os.path.join(args.save_dir, data_filename), datum)
    csv_filename = f'{args.data_file_base_name}_{args.current_seed}.csv'
    with open(os.path.join(args.save_dir, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')


def get_data_path_names(args):
    training_data_files = glob.glob(os.path.join(args.data_csv_dir, "*training*.csv"))
    testing_data_files = glob.glob(os.path.join(args.data_csv_dir, "*testing*.csv"))
    return training_data_files, testing_data_files


def image_aligned_to_non_subgoal(image, r_pose, vertex_point):
    """Permutes an image from axis-aligned to subgoal-pointing frame.
    The subgoal should appear at the center of the image."""
    cols = image.shape[1]
    sp = vertex_point
    yaw = np.arctan2(sp[1] - r_pose.y, sp[0] - r_pose.x) - r_pose.yaw
    roll_amount = int(round(-cols * yaw / (2 * math.pi)))
    return np.roll(image, shift=roll_amount, axis=1)


def get_rel_goal_loc_vecs(pose, goal_pose, num_bearing, vertex_point=None):
    # Lookup vectors
    _, vec_bearing = lsp.utils.learning_vision.get_directions(num_bearing)
    if vertex_point is None:
        vec_bearing = vec_bearing + pose.yaw
    else:
        sp = vertex_point
        vertex_point_yaw = np.arctan2(sp[1] - pose.y, sp[0] - pose.x)
        vec_bearing = vec_bearing + vertex_point_yaw

    robot_point = np.array([pose.x, pose.y])
    goal_point = np.array([goal_pose.x, goal_pose.y])
    rel_goal_point = goal_point - robot_point

    goal_loc_x_vec = rel_goal_point[0] * np.cos(
        vec_bearing) + rel_goal_point[1] * np.sin(vec_bearing)
    goal_loc_y_vec = -rel_goal_point[0] * np.sin(
        vec_bearing) + rel_goal_point[1] * np.cos(vec_bearing)

    return (goal_loc_x_vec[:, np.newaxis].T, goal_loc_y_vec[:, np.newaxis].T)


def get_oriented_non_subgoal_input_data(pano_image, robot_pose, goal_pose, vertex_point):
    """Helper function that returns a dictionary of the input data provided to the
neural network in the 'oriented' data configuration. The 'pano_image' is assumed
to be in the robot coordinate frame, and will be 'rotated' such that the subgoal
of interest is at the center of the image. Similarly, the goal information will
be stored as two vectors with each element corresponding to the sin and cos of
the relative position of the goal in the 'oriented' image frame."""

    # Re-orient the image based on the subgoal centroid
    oriented_pano_image = image_aligned_to_non_subgoal(
        pano_image, robot_pose, vertex_point)

    # Compute the goal information
    num_bearing = pano_image.shape[1] // 4
    goal_loc_x_vec, goal_loc_y_vec = get_rel_goal_loc_vecs(
        robot_pose, goal_pose, num_bearing, vertex_point)

    # sc = subgoal.get_centroid()
    sc = vertex_point
    subgoal_pose = common.Pose(sc[0], sc[1], 0)
    subgoal_loc_x_vec, subgoal_loc_y_vec = get_rel_goal_loc_vecs(
        robot_pose, subgoal_pose, num_bearing, vertex_point)

    return {
        'image': oriented_pano_image,
        'goal_loc_x': goal_loc_x_vec,
        'goal_loc_y': goal_loc_y_vec,
        'subgoal_loc_x': subgoal_loc_x_vec,
        'subgoal_loc_y': subgoal_loc_y_vec,
    }


def get_path_middle_point(known_map, start, goal, args):
    """This function returns the middle point on the path from goal to the
    robot starting position"""
    inflation_radius = args.inflation_radius_m / args.base_resolution
    inflated_mask = gridmap.utils.inflate_grid(known_map,
                                               inflation_radius=inflation_radius)
    # Now sample the middle point
    cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
        inflated_mask, [goal.x, goal.y])
    _, path = get_path([start.x, start.y],
                       do_sparsify=False,
                       do_flip=False)
    row, col = path.shape
    x = path[0][col // 2]
    y = path[1][col // 2]
    new_start_pose = common.Pose(x=x,
                                 y=y,
                                 yaw=2 * np.pi * np.random.rand())
    return new_start_pose


def parse_args():
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--relative_positive_weight',
                        default=1.0,
                        help='Initial learning rate',
                        type=float)
    parser.add_argument('--do_randomize_start_pose', action='store_true')
    parser.add_argument(
        '--current_seed', 
        type=int)
    parser.add_argument(
        '--data_file_base_name', 
        type=str, 
        required=False)
    parser.add_argument('--logfile_name', type=str, default='logfile.txt')
    group = parser.add_argument_group('Make Training Data Arguments')
    group.add_argument(
        '--network_file',
        type=str,
        required=False,
        help='Directory with the name of the saved model')
    group.add_argument(
        '--autoencoder_network_file',
        type=str,
        required=False,
        help='Directory with the name of the autoencoder model')
    group.add_argument(
        '--image_filename',
        type=str,
        required=False,
        help='File name for the completed evaluations')
    group.add_argument(
        '--data_csv_dir',
        type=str,
        required=False,
        help='Directory in which to save the data csv')
    group.add_argument(
        '--pickle_directory',
        type=str,
        required=False,
        help='Directory in which to save the pickle dataums')
    group.add_argument(
        '--csv_basename',
        type=str,
        required=False,
        help='Directory in which to save the CSV base file')
    group = parser.add_argument_group('Neural Network Training Testing \
        Arguments')
    group.add_argument(
        '--core_directory',
        type=str,
        required=False,
        help='Directory in which to look for data')
    group.add_argument(
        '--num_training_elements',
        type=int,
        required=False,
        default=5000,
        help='Number of training samples')
    group.add_argument(
        '--num_testing_elements',
        type=int,
        required=False,
        default=1000,
        help='Number of testing samples')
    group.add_argument(
        '--num_steps',
        type=int,
        required=False,
        default=10000,
        help='Number of steps while iterating')
    group.add_argument(
        '--test_log_frequency',
        type=int,
        required=False,
        default=10,
        help='Frequecy of testing log to be generated')
    group.add_argument(
        '--learning_rate',
        type=float,
        required=False,
        default=.001,
        help='Learning rate of the model')
    group.add_argument('--experiment_name', type=str, default='base')

    return parser.parse_args()
