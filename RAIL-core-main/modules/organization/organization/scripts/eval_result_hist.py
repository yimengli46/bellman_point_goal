import argparse
import matplotlib.pyplot as plt
import pandas as pd
import re


def process_results_data(args):
    # Load data and loop through rows
    data = []
    for line in open(args.data_file).readlines():
        d = re.match(r".*?s: (.*?) . learned: (.*?) . baseline: (.*?)\n", line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1]), float(d[2])])

    return pd.DataFrame(data, columns=["seed", "learned_cost", "baseline_cost"])


"""
Compare expected cost from learned solver and non learned solver using non learned solver
to evaluate the learned cost and plot it on the histogram

"""


def build_plot(fig, data, args):
    diff_exp_costs = abs(data["learned_cost"] - data["baseline_cost"]).to_list()
    fig.gca()
    plt.subplot(111)
    plt.hist(diff_exp_costs, density=False, edgecolor="black")
    plt.locator_params(axis="y", integer=True)
    plt.ylabel("Number of env")
    plt.xlabel("Difference in Optimal Expected Costs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a figure for evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_file", type=str, required=False, default=None)
    parser.add_argument("--output_image_file", type=str, required=False, default=None)
    parser.add_argument("--xpassthrough", type=str, required=False, default="false")
    args = parser.parse_args()

    data = process_results_data(args)

    # Below 10000
    print(data.describe())
    fig = plt.figure(dpi=300, figsize=(5, 5))
    build_plot(fig, data, args)
    plt.savefig(args.output_image_file, dpi=300)
