import organization
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
import os
import learning
from examples.continuous_tamp.primitives import draw_state, SUCTION_HEIGHT
from examples.continuous_tamp.viewer import ContinuousTMPViewer
from collections import namedtuple
import argparse
import random


TAMPState = namedtuple("TAMPState", ["robot_confs", "holding", "block_poses"])

robot_conf = organization.environments.blockworld.INITIAL_CONF
ContinuousTMPViewer.draw_region = _draw_region
ContinuousTMPViewer.draw_block = _draw_block

"""
Generate blockworld environment per seed and saved it on the image along with the
expected cost.
"""


def data_gen_blockworld(args, seed):
    print(f"Generating new world (seed: {seed})")
    regions = generate_regions(seed)
    environment = organization.environments.blockworld.BlockworldEnvironment(
        regions=regions, verbose=True
    )

    random.seed(seed)
    region_poses = get_region_dims(regions, args.num_blocks)
    task_distribution = gen_simple_task_distribution()

    for i in range(len(region_poses)):
        block_state = gen_block_states_based_on_regions(region_poses[i])
        colors = dict(zip(sorted(block_state.keys()), COLORS))
        viewer = ContinuousTMPViewer(SUCTION_HEIGHT, regions, title="Continuous TAMP")
        state = TAMPState(
            robot_confs={"r0": robot_conf}, holding={}, block_poses=block_state
        )
        draw_state(viewer, state, colors)
        expected_cost_state = organization.core.get_expected_cost(
            environment, block_state, task_distribution
        )

        if str(expected_cost_state) != "inf":
            datum = {
                "image": return_image(viewer),
                "expected_cost": expected_cost_state,
            }
        else:
            continue

        if datum is None:
            continue

        write_datum_to_pickle(args, i, datum)
        print(f"Saved state: {args.seed}.{i}")
    # Write a final file that indicates training is done
    # need to work here

    block_state = gen_block_states_based_on_regions(region_poses[0])
    state = TAMPState(
        robot_confs={"r0": robot_conf}, holding={}, block_poses=block_state
    )
    draw_state(viewer, state, colors)
    viewer.save(
        os.path.join(
            args.base_data_path,
            "data",
            "training_env_plots",
            f"{args.data_plot_name}_{args.seed}",
        )
    )


def write_datum_to_pickle(args, counter, datum):
    save_dir = os.path.join(args.base_data_path, "data")
    data_filename = os.path.join("pickles", f"dat_{args.seed}_{counter}.pgz")
    learning.data.write_compressed_pickle(os.path.join(save_dir, data_filename), datum)

    csv_filename = f"{args.data_file_base_name}_{args.seed}.csv"
    with open(os.path.join(save_dir, csv_filename), "a") as f:
        f.write(f"{data_filename}\n")


def get_parser():
    parser = argparse.ArgumentParser(description="Generate blockworld env data")
    parser.add_argument("--base_data_path", type=str, default="/data/")
    parser.add_argument("--data_file_base_name", type=str, required=True)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--data_plot_name", type=str, required=True)
    parser.add_argument("--seed", type=int)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    data_gen_blockworld(args, args.seed)
