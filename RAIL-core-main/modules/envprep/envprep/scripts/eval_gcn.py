import random
import matplotlib.pyplot as plt
import os
import envprep
import organization
from envprep.utilities.gcn_data_gen_helper import (
    get_nodes_from_curr_state,
    generate_regions,
    get_all_regions_for_blocks,
    get_block_states,
    get_edge_features,
    robot_conf,
    graph_format,
)
from envprep.utilities.gcn_eval_helper import (
    write_all_in_one_file,
    write_csv_and_return_image,
    get_parser,
)
from envprep.models import prepare_env_gcn
import torch


def optimal_expected_cost_from_pddlstream(args, regions, blocks_poses, obstacles):
    """
    Get optimal expected cost using non learned solver. Takes in task distribution

    Args: environment,regions
    Return :
    Image and expected cost per state.
    """
    state_dict = {"state": "non_learned_expected_cost"}
    task_distribution = [[0.8, [("A", "red")]], [0.2, [("A", "blue")]]]
    environment = envprep.environments.blockworld2D.BlockworldEnvironment(
        regions=regions, obstacles=obstacles, verbose=True
    )
    for i in range(len(blocks_poses)):
        block_state = get_block_states(blocks_poses[i])
        expected_cost_state = organization.core.get_expected_cost(
            environment, block_state, task_distribution
        )
        if expected_cost_state is None:
            return None, None

        state_dict.update({str(blocks_poses[i]): expected_cost_state})

    img, e_cost = write_csv_and_return_image(
        args, state_dict, regions, obstacles, "non_learned"
    )

    return img, e_cost


def optimal_expected_cost_from_learning(args, regions, blocks_poses, obstacles):
    """
    Get optimal expected cost using learned solver. Doesn't take task distribution.

    Args: regions
    Return: Image and expected cost per state.

    """
    state_dict = {"state": "learned_expected_cost"}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_net = prepare_env_gcn.PrepareEnvGCN.get_net_eval_fn(
        network_file=args.network_file, device=device
    )

    for i in range(len(blocks_poses)):
        block_state = get_block_states(blocks_poses[i])
        nodes_arr = get_nodes_from_curr_state(block_state, regions, robot_conf)
        nodes = graph_format(nodes_arr)
        node_vals = []
        for node in nodes_arr:
            if {} in list(node.values()):
                continue
            node_vals.extend(list(node.values()))

        edge_features, edge_index = get_edge_features(node_vals)

        datum = {
            "graph_nodes": torch.tensor(nodes, dtype=torch.float),
            "graph_edge_index": torch.tensor(edge_index, dtype=torch.long),
            "graph_edge_feats": torch.tensor(edge_features, dtype=torch.float),
            "num_nodes": len(nodes),
        }
        expected_cost_state = eval_net(datum)
        state_dict.update({str(blocks_poses[i]): expected_cost_state})

    img, e_cost = write_csv_and_return_image(
        args, state_dict, regions, obstacles, "learned"
    )

    return img, e_cost


def results_to_file(args, seed):
    """
    Save image and expected cost from both learned and non learned solver to the file.
    """
    regions = generate_regions(seed)
    blocks_poses = get_all_regions_for_blocks(regions, args.num_blocks)
    obstacles = []
    random.seed(seed)

    learned_state, learned_cost = optimal_expected_cost_from_learning(
        args, regions, blocks_poses, obstacles
    )

    non_learned_state, non_learned_cost = optimal_expected_cost_from_pddlstream(
        args, regions, blocks_poses, obstacles
    )

    if non_learned_cost is None:
        return None

    fig = plt.figure()

    fig.add_subplot(221)
    plt.axis("off")
    plt.title("s*_GNN with C_GNN: " + str(round(learned_cost, 4)), fontsize=10)
    plt.imshow(learned_state)

    # show original image
    fig.add_subplot(222)
    plt.axis("off")
    plt.title("s*_PDDL with C_PDDL: " + str(round(non_learned_cost, 4)), fontsize=10)
    plt.imshow(non_learned_state)

    plt.savefig(
        os.path.join(args.base_results_path, f"prepared_state_env_{args.seed}.png")
    )
    success = 1
    return success


if __name__ == "__main__":
    args = get_parser().parse_args()
    valid_val = results_to_file(args, args.seed)
    if valid_val is not None:
        write_all_in_one_file(args)
