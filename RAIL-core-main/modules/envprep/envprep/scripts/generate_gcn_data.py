import organization
import envprep
from organization.scripts.generate_data import write_datum_to_pickle
from envprep.utilities.gcn_data_gen_helper import (
    get_nodes_from_curr_state,
    generate_regions,
    get_all_regions_for_blocks,
    get_block_states,
    COLORS,
    get_edge_features,
    graph_format,
    robot_conf,
)
from examples.continuous_tamp.primitives import TAMPState
import os
import argparse
import random
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from PIL import Image
from envprep.environments.blockworld.viewer import ContinuousTMPViewer


"""
Generate blockworld environment per seed and saved it on the graph that has the
expected cost as meta node.
"""


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img


def return_graph(nodes, edge_index, node_names, color_map):
    plt.close()
    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    G = to_networkx(data, to_undirected=True)
    nx.draw(G, labels=node_names, node_color=color_map)
    fig = plt.gcf()
    img = fig2img(fig)
    return img


def images_together(graph_image, blockworld_image):
    # plt.clf()
    fig1 = plt.figure(figsize=(10, 7))

    # showing image
    fig1.add_subplot(1, 2, 1)
    plt.imshow(blockworld_image)
    plt.axis("off")
    plt.title("Blockworld")
    # Adds a subplot at the 2nd position
    fig1.add_subplot(1, 2, 2)

    # showing image
    plt.imshow(graph_image)
    plt.axis("off")
    plt.title("Graph of Blockworld")


def gcn_data_gen_blockworld(args, seed):
    print(f"Generating new world (seed: {seed})")
    regions = generate_regions(seed)
    # obstacles = generate_obstacles(seed)
    obstacles = []
    nodes = []
    edge_index = []
    environment = envprep.environments.blockworld2D.BlockworldEnvironment(
        regions=regions, obstacles=obstacles, verbose=True
    )

    random.seed(seed)
    blocks_poses = get_all_regions_for_blocks(regions, args.num_blocks)
    task_distribution = [[0.8, [("A", "red")]], [0.2, [("A", "blue")]]]
    for i in range(len(blocks_poses)):
        block_state = get_block_states(blocks_poses[i])
        expected_cost_state = organization.core.get_expected_cost(
            environment, block_state, task_distribution
        )

        if expected_cost_state is None:
            return

        nodes_arr = get_nodes_from_curr_state(block_state, regions, robot_conf)
        nodes = graph_format(nodes_arr)
        node_names = []
        node_vals = []
        for node in nodes_arr:
            if {} in list(node.values()):
                continue
            node_vals.extend(list(node.values()))
            node_names.extend(list(node.keys()))

        node_names = {i: node_names[i] for i in range(len(node_names))}
        color_map = [feats["color"] for feats in node_vals]
        edge_features, edge_index = get_edge_features(node_vals)
        image = return_graph(nodes, edge_index, node_names, color_map)
        if str(expected_cost_state) != "inf":
            datum = {
                "graph_nodes": torch.tensor(nodes, dtype=torch.float),
                "graph_edge_index": torch.tensor(edge_index, dtype=torch.long),
                "graph_edge_feats": torch.tensor(edge_features, dtype=torch.float),
                "graph_image": image,
                "expected_cost": expected_cost_state,
            }
        else:
            continue

        if datum is None:
            continue

        write_datum_to_pickle(args, i, datum)
        print(f"Saved state: {args.seed}.{i}")

    colors = dict(zip(sorted(block_state.keys()), COLORS))
    viewer = ContinuousTMPViewer(regions, obstacles)
    block_state = get_block_states(blocks_poses[0])
    state = TAMPState(
        robot_confs={"r0": robot_conf}, holding={}, block_poses=block_state
    )
    viewer.draw_state(state, colors)
    plt.clf()
    original_img = viewer.return_image()
    graph_img = return_graph(nodes, edge_index, node_names, color_map)
    images_together(graph_img, original_img)
    plt.savefig(
        os.path.join(
            args.base_data_path,
            "data",
            "training_env_plots",
            f"{args.data_plot_name}_{args.seed}",
        )
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Generate blockworld env data")
    parser.add_argument("--base_data_path", type=str, default="/data/")
    parser.add_argument("--data_file_base_name", type=str, required=True)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--data_plot_name", type=str, required=True)
    parser.add_argument("--seed", type=int)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    gcn_data_gen_blockworld(args, args.seed)
