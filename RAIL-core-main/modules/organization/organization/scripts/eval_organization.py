import organization
import argparse
import os
from examples.continuous_tamp.viewer import ContinuousTMPViewer
from organization.utilities.eval_helper import write_all_in_one_file
from organization.utilities.bw_data_gen_helper import (
    generate_regions,
    return_image,
    _draw_region,
    COLORS,
    gen_simple_task_distribution,
    _draw_block,
    gen_block_states_based_on_regions,
    get_region_dims,
)
from organization.models import organizationCNN
import torch
import csv
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from examples.continuous_tamp.primitives import draw_state, SUCTION_HEIGHT

TAMPState = namedtuple("TAMPState", ["robot_confs", "holding", "block_poses"])
robot_conf = organization.environments.blockworld.INITIAL_CONF
ContinuousTMPViewer.draw_region = _draw_region
ContinuousTMPViewer.draw_block = _draw_block


def write_csv_and_return_image(args, seed, state_dict, regions, f_name):
    """
    Args: arguments, seed as in environment number, state dictionary (poses and expected cost), regions
    and file name.

    Return :
    write state and corresponding expected cost to csv file and
    return image and state dictionary.
    """

    save_dir = os.path.join(args.base_results_path)
    data_filename = os.path.join(f_name, f"{f_name}_env_{args.seed}.csv")
    with open(os.path.join(save_dir, data_filename), "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in state_dict.items():
            writer.writerow([key, value])

    del state_dict["state"]
    optimal_region_pos = min(state_dict, key=state_dict.get)
    # find optimal_region_pos here
    optimal_state = gen_block_states_based_on_regions(optimal_region_pos)
    colors = dict(zip(sorted(optimal_state.keys()), COLORS))
    opt_view = ContinuousTMPViewer(SUCTION_HEIGHT, regions, title="Continuous_TAMP")
    state = TAMPState(
        robot_confs={"r0": robot_conf}, holding={}, block_poses=optimal_state
    )
    draw_state(opt_view, state, colors)

    return return_image(opt_view), state_dict[optimal_region_pos]


def optimal_expected_cost_from_pddlstream(environment, regions, args, seed):
    """
    Get optimal expected cost using non learned solver. Takes in task distribution

    Args: environment,regions
    Return :
    Image and expected cost per state.
    """

    state_dict = {"state": "non_learned_expected_cost"}
    region_poses = get_region_dims(regions, args.num_blocks)
    task_distribution = gen_simple_task_distribution()

    for i in range(len(region_poses)):
        block_state = gen_block_states_based_on_regions(region_poses[i])
        expected_cost_state = organization.core.get_expected_cost(
            environment, block_state, task_distribution
        )
        state_dict.update({region_poses[i]: expected_cost_state})

    img, e_cost = write_csv_and_return_image(
        args, seed, state_dict, regions, "non_learned"
    )

    return img, e_cost


def optimal_expected_cost_from_learning(regions, args, seed):
    """
    Get optimal expected cost using learned solver. Doesn't take task distribution.

    Args: regions
    Return :
    Image and expected cost per state.

    """

    state_dict = {"state": "learned_expected_cost"}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_net = organizationCNN.OrganizationNet.get_net_eval_fn(
        network_file=args.network_file, device=device
    )

    region_poses = get_region_dims(regions, args.num_blocks)

    for i in range(len(region_poses)):
        block_state = gen_block_states_based_on_regions(region_poses[i])
        colors = dict(zip(sorted(block_state.keys()), COLORS))
        view = ContinuousTMPViewer(SUCTION_HEIGHT, regions, title="Continuous TAMP")
        state = TAMPState(
            robot_confs={"r0": robot_conf}, holding={}, block_poses=block_state
        )
        draw_state(view, state, colors)
        expected_cost_state = eval_net(return_image(view))
        state_dict.update({region_poses[i]: expected_cost_state})

    img, e_cost = write_csv_and_return_image(args, seed, state_dict, regions, "learned")

    return img, e_cost


def results_to_file(args, seed):
    """
    Save image and expected cost from both learned and non learned solver to the file.
    """

    regions = generate_regions(seed)
    random.seed(seed)
    environment = organization.environments.blockworld.BlockworldEnvironment(
        regions=regions, verbose=True
    )

    non_learned_state, non_learned_cost = optimal_expected_cost_from_pddlstream(
        environment, regions, args, seed
    )
    learned_state, learned_cost = optimal_expected_cost_from_learning(
        regions, args, seed
    )
    fig = plt.figure()

    fig.add_subplot(221)
    plt.axis("off")
    plt.title("learned_expected_cost: " + str(round(learned_cost, 4)), fontsize=10)
    plt.imshow(learned_state)

    # show original image
    fig.add_subplot(222)
    plt.axis("off")
    plt.title(
        "non_learned_expected_cost: " + str(round(non_learned_cost, 4)), fontsize=10
    )
    plt.imshow(non_learned_state)

    plt.savefig(
        os.path.join(args.base_results_path, f"optimal_state_env_{args.seed}.png")
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluating the model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--network_file", type=str)
    parser.add_argument("--base_results_path", type=str, default="/results/")
    parser.add_argument("--eval_folder", type=str, default="/learned/")
    parser.add_argument("--log_file", type=str, default="logfile.txt")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    results_to_file(args, args.seed)
    write_all_in_one_file(args)
