import os
import argparse
import pandas as pd
import numpy as np
import csv
from examples.continuous_tamp.primitives import TAMPState
from envprep.environments.blockworld.viewer import ContinuousTMPViewer
from envprep.utilities.gcn_data_gen_helper import get_block_states, COLORS, robot_conf


def str_to_tuple(str):
    str = str.replace("array", "")
    str = str.replace(" ", "")
    tup = eval(str)
    return (np.array(tup[0]), np.array(tup[1]))


def write_csv_and_return_image(args, state_dict, regions, obstacles, f_name):
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
    optimal_region_pos_str = min(state_dict, key=state_dict.get)
    optimal_region_pos = str_to_tuple(optimal_region_pos_str)
    # find optimal_region_pos here
    optimal_state = get_block_states(optimal_region_pos)
    colors = dict(zip(sorted(optimal_state.keys()), COLORS))
    viewer = ContinuousTMPViewer(regions, obstacles)
    state = TAMPState(
        robot_confs={"r0": robot_conf}, holding={}, block_poses=optimal_state
    )
    viewer.draw_state(state, colors)

    return viewer.return_image(), state_dict[optimal_region_pos_str]


def write_all_in_one_file(args):
    """
    Compare expected cost from learned solver and non learned solver and save the costs and
    states in csv file.
    """

    learned_folder = os.path.join(args.eval_folder, "learned")
    non_learned_folder = os.path.join(args.eval_folder, "non_learned")
    learned_csv = os.path.join(learned_folder, f"learned_env_{args.seed}.csv")
    non_learned_csv = os.path.join(
        non_learned_folder, f"non_learned_env_{args.seed}.csv"
    )
    learned_df = pd.read_csv(learned_csv)
    non_learned_df = pd.read_csv(non_learned_csv)
    C_PDDL_Sstar_PDDL = non_learned_df["non_learned_expected_cost"].min()
    non_learned_opt_state_index = non_learned_df["non_learned_expected_cost"].idxmin()
    learned_opt_state_index = learned_df["learned_expected_cost"].idxmin()
    non_learned_opt_state = non_learned_df.iloc[non_learned_opt_state_index]["state"]
    learned_opt_state = learned_df.iloc[learned_opt_state_index]["state"]
    C_PDDL_Sstar_GNN = non_learned_df.iloc[learned_opt_state_index][
        "non_learned_expected_cost"
    ]

    statesfile = os.path.join(args.states_file)
    with open(statesfile, "a+") as f:
        f.write(
            f"[Learn] s: {args.seed:4d}"
            f" | best_state_GNN: {learned_opt_state}"
            f" | best_state_PDDL: {non_learned_opt_state}\n"
        )

    logfile = os.path.join(args.log_file)
    with open(logfile, "a+") as f:
        f.write(
            f"[Learn] s: {args.seed:4d}"
            f" | learned_GNN: {C_PDDL_Sstar_GNN:0.3f}"
            f" | baseline_PDDL: {C_PDDL_Sstar_PDDL:0.3f}\n"
        )


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluating the model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--network_file", type=str)
    parser.add_argument("--base_results_path", type=str, default="/results/")
    parser.add_argument("--eval_folder", type=str, default="/learned/")
    parser.add_argument("--log_file", type=str, default="logfile.txt")
    parser.add_argument("--states_file", type=str, default="statesfile.txt")

    return parser
