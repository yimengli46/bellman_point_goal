import string
from itertools import permutations
import random
import numpy as np
import envprep
from math import sqrt
from matplotlib import colors

COLORS = ["red", "orange", "yellow", "green", "blue", "violet", "white", "black"]
REGIONS = ["yellow", "green", "violet", "white"]
robot_conf = envprep.environments.blockworld2D.INITIAL_CONF


def generate_regions(env_num):
    """
    Parameters
    ----------
    env_num : environment number which is a seed for random generation of
    regions in the environment

    Returns
    -------
    regions: randomly generate regions for current environment (env_num)

    """
    random.seed(env_num)
    num_of_regions = random.randint(0, 3)
    regions_sampled = random.sample(REGIONS, num_of_regions)
    regions_sampled.append("red")
    regions_sampled.append("blue")
    random.shuffle(regions_sampled)
    possible_regions_dims = [
        [(0, 0), (100, 100)],
        [(200, 0), (300, 100)],
        [(300, 0), (400, 100)],
        [(0, 300), (100, 400)],
        [(200, 300), (300, 400)],
        [(300, 300), (400, 400)],
        [(0, 100), (100, 200)],
        [(300, 100), (400, 200)],
    ]

    pos_required = len(regions_sampled)
    poses = random.sample(possible_regions_dims, pos_required)
    regions_poses = [poses[i] for i in range(len(regions_sampled))]
    regions_sampled.insert(0, "grey")
    regions_poses.insert(0, [(0, 0), (400, 400)])
    regions = dict(zip(regions_sampled, regions_poses))

    return regions


def generate_obstacles(env_num):
    """
    Parameters
    ----------
    env_num : environment number which is a seed for random generation of
    regions in the environment

    Returns
    -------
    robstacles: randomly generate obstacles for current environment (env_num)

    """
    random.seed(env_num)
    num_obstacles = random.randint(0, 2)
    possible_obstacles = [
        [(0, 205), (130, 210)],
        [(190, 0), (195, 140)],
        [(190, 280), (195, 400)],
        [(270, 205), (400, 210)],
    ]
    obstacles_in_env = random.sample(possible_obstacles, num_obstacles)
    return obstacles_in_env


def get_task_distribution(num_blocks):
    """
    Parameters
    ----------
    num_blocks : number of blocks in the environment. It gives us an idea of
    potential number of tasks in the environment.

    Returns
    -------
    task_distribution: hand coded task distribution based on number of blocks.

    """
    if num_blocks == 1:
        num_tasks = random.randint(1, 2)
        if num_tasks == 1:
            task_distribution = [[1, [("A", "red")]]]
        else:
            tasks_prob = np.random.dirichlet(np.ones(num_tasks), size=1).tolist()[0]
            task_distribution = [
                [tasks_prob[0], [("A", "red")]],
                [tasks_prob[1], [("A", "blue")]],
            ]

    if num_blocks >= 2:
        num_tasks = random.randint(1, 4)
        if num_tasks == 1:
            task_distribution = [[1, [("A", "red")]]]
        elif num_tasks == 2:
            tasks_prob = np.random.dirichlet(np.ones(num_tasks), size=1).tolist()[0]
            second_block = random.choice(["A", "B"])
            task_distribution = [
                [tasks_prob[0], [("A", "red")]],
                [tasks_prob[1], [(second_block, "blue")]],
            ]
        elif num_tasks == 3:
            tasks_prob = np.random.dirichlet(np.ones(num_tasks), size=1).tolist()[0]
            task_distribution = [
                [tasks_prob[0], [("A", "red")]],
                [tasks_prob[1], [("A", "blue")]],
                [tasks_prob[2], [("B", "blue")]],
            ]
        else:
            tasks_prob = np.random.dirichlet(np.ones(num_tasks), size=1).tolist()[0]
            task_distribution = [
                [tasks_prob[0], [("A", "red")]],
                [tasks_prob[1], [("A", "blue")]],
                [tasks_prob[2], [("B", "blue")]],
                [tasks_prob[3], [("B", "red")]],
            ]

    return task_distribution


def get_all_regions_for_blocks(regions, num_blocks):
    """
    Parameters
    ----------
    regions : regions in the environment
    num_blocks :number of blocks in the environment.

    Returns
    -------
    regions: all the random region poses where blocks can be placed in
    as the states based on regions.

    """
    region_vals = list(regions.values())
    positions = []
    for region in region_vals:
        if region == [(0, 0), (400, 400)]:
            region = [(100, 100), (300, 300)]
        lower, upper = region
        lower_x, lower_y = lower
        upper_x, upper_y = upper
        x_pose = np.random.uniform(lower_x + 40, upper_x - 40)
        y_pose = np.random.uniform(lower_y + 40, upper_y - 40)
        block_pos = np.array([x_pose, y_pose])
        positions.append(block_pos)

    regions_for_blocks = list(permutations(positions, num_blocks))
    return regions_for_blocks


def get_block_states(regions):
    """
    Parameters
    ----------
    regions : regions in the environment

    Returns
    -------
    initial_block_poses: random initial block poses

    """
    num_blocks = len(regions)
    blocks = [string.ascii_uppercase[j] for j in range(num_blocks)]
    block_pos_tuple_list = []
    region_pos = list(regions)
    block_pos_tuple_list = [region_pos[i] for i in range(len(blocks))]

    initial_block_poses = dict(zip(blocks, block_pos_tuple_list))
    return initial_block_poses


def get_object_type_encoding(object):
    encoding = [0, 0, 0, 0]
    objects = ["block", "region", "obstacle", "robot"]
    object_indx = objects.index(object)
    encoding[object_indx] = 1
    return encoding


def distance_between(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_edge_features(nodes):
    perm = permutations([i for i in range(len(nodes))], 2)
    edges = list(perm)
    source = [s[0] for s in edges]
    dest = [s[1] for s in edges]
    edge_index = [source, dest]
    features = []
    for node in nodes:
        for node2 in nodes:
            if node == node2:
                continue
            position_node_1 = node["position"]
            point1_inds = [i for i, e in enumerate(position_node_1) if e != 0]
            point1 = (position_node_1[point1_inds[0]], position_node_1[point1_inds[1]])

            position_node_2 = node2["position"]
            point2_inds = [i for i, e in enumerate(position_node_2) if e != 0]
            point2 = (position_node_2[point2_inds[0]], position_node_2[point2_inds[1]])

            features.append([distance_between(point1, point2)])

    return features, edge_index


def get_region_center(region_dims):
    return (
        (region_dims[0][0] + region_dims[1][0]) / 2,
        (region_dims[0][1] + region_dims[1][1]) / 2,
    )


def get_block_nodes(blocks):
    blocks_names = list(blocks.keys())
    block_poses = list(blocks.values())
    parent, curr_node, block_nodes = {}, {}, []
    for i in range(len(blocks_names)):
        curr_node["type"] = get_object_type_encoding("block")  # class
        curr_node["color"] = COLORS[i]  # color
        curr_node["position"] = [
            block_poses[i][0] / 400,
            block_poses[i][1] / 400,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # position
        parent[str(blocks_names[i])] = curr_node
        curr_node = {}

    block_nodes.append(parent)

    return block_nodes


def get_region_nodes(regions):
    regions_names = list(regions.keys())
    regions_dims = list(regions.values())
    parent, curr_node, region_nodes = {}, {}, []
    for i in range(len(regions_names)):
        if regions_names[i] == "grey":
            continue
        curr_node["type"] = get_object_type_encoding("region")  # class
        curr_node["color"] = regions_names[i]  # color
        region_center = get_region_center(regions_dims[i])
        curr_node["position"] = [
            0,
            0,
            region_center[0] / 400,
            region_center[1] / 400,
            0,
            0,
            0,
            0,
        ]  # position
        parent[str(regions_names[i])] = curr_node
        curr_node = {}

    region_nodes.append(parent)

    return region_nodes


def get_obstacle_nodes(obstacles):
    parent, curr_node, obstacle_nodes = {}, {}, []
    if len(obstacles) == 0:
        parent["obstacle"] = {}
        return [parent]

    for i in range(len(obstacles)):
        curr_node["type"] = get_object_type_encoding("obstacle")  # class
        curr_node["color"] = "black"  # color
        obs_center = get_region_center(obstacles[i])
        curr_node["position"] = [
            0,
            0,
            0,
            0,
            obs_center[0] / 400,
            obs_center[1] / 400,
            0,
            0,
        ]  # position
        parent["obstacle"] = curr_node
        curr_node = {}

    obstacle_nodes.append(parent)

    return obstacle_nodes


def get_robot_node(robot_conf):
    parent, curr_node, robot_node = {}, {}, []
    curr_node["type"] = get_object_type_encoding("robot")  # class
    curr_node["color"] = "yellow"  # color
    curr_node["position"] = [
        0,
        0,
        0,
        0,
        0,
        0,
        robot_conf[0] / 400,
        robot_conf[1] / 400,
    ]  # position
    parent["robot"] = curr_node
    robot_node.append(parent)

    return robot_node


def get_nodes_from_curr_state(blocks, regions, robot_conf, obstacles=[]):
    block_nodes = get_block_nodes(blocks)
    region_nodes = get_region_nodes(regions)
    obstacle_nodes = get_obstacle_nodes(obstacles)
    robot_node = get_robot_node(robot_conf)
    all_nodes = block_nodes + region_nodes + obstacle_nodes + robot_node

    return all_nodes


def graph_format(nodes):
    node_names = []
    node_vals = []
    for node in nodes:
        node_names.extend(list(node.keys()))
        node_vals.extend(list(node.values()))

    val_arr = []
    for vals in node_vals:
        if vals == {}:
            continue
        color = vals["color"]
        vals["color"] = list(colors.to_rgba(color))
        val_arr.append(list(vals.values()))

    graph_array = []
    for node in val_arr:
        node_feat = [item for feat in node for item in feat]
        graph_array.append(node_feat)

    return graph_array
