import organization
import os
import pandas as pd
from organization.utilities.bw_data_gen_helper import (
    generate_regions,
    gen_simple_task_distribution,
    gen_block_states_based_on_regions,
)


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
    non_learned_opt_expected_cost = non_learned_df["non_learned_expected_cost"].min()

    learned_opt_state_index = learned_df["learned_expected_cost"].idxmin()
    learned_opt_state = learned_df.iloc[learned_opt_state_index]["state"]
    regions = generate_regions(args.seed)
    task_distribution = gen_simple_task_distribution()
    # find region_pos here
    learned_opt_state = eval(learned_opt_state)
    block_state = gen_block_states_based_on_regions(learned_opt_state)
    environment = organization.environments.blockworld.BlockworldEnvironment(
        regions=regions, verbose=True
    )
    expected_cost_learned_opt_state = organization.core.get_expected_cost(
        environment, block_state, task_distribution
    )

    logfile = os.path.join(args.log_file)
    with open(logfile, "a+") as f:
        f.write(
            f"[Learn] s: {args.seed:4d}"
            f" | learned: {expected_cost_learned_opt_state:0.3f}"
            f" | baseline: {non_learned_opt_expected_cost:0.3f}\n"
        )
