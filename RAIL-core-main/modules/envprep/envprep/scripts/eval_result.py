import argparse
import matplotlib.pyplot as plt
import pandas as pd
import re


def process_results_data(args):
    # Load data and loop through rows
    data = []
    for line in open(args.data_file).readlines():
        d = re.match(r".*?s: (.*?) . learned_GNN: (.*?) . baseline_PDDL: (.*?)\n", line)
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
    corr_count = 0
    for exp_costs in diff_exp_costs:
        if exp_costs < 50:
            corr_count += 1

    incorr_count = len(diff_exp_costs) - corr_count
    fig.gca()
    plt.subplot(111)
    plt.pie(
        [corr_count, incorr_count],
        autopct=lambda p: "{:.0f}".format(p * len(diff_exp_costs) / 100),
    )
    labels = ["correct prepared states", "incorrect prepared states"]
    patches, _ = plt.pie([corr_count, incorr_count], startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.savefig(args.pie_chart, dpi=400)

    plt.clf()
    plt.subplot(111)
    plt.hist(diff_exp_costs, density=False, edgecolor="black")
    plt.locator_params(axis="y", integer=True)
    plt.ylabel("Number of environments")
    plt.xlabel("C_PDDL(s*_PDDL) - C_PDDL(s*_GCN)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a figure for evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_file", type=str, required=False, default=None)
    parser.add_argument("--states_file", type=str, required=False, default=None)
    parser.add_argument("--output_image_file", type=str, required=False, default=None)
    parser.add_argument("--pie_chart", type=str, required=False, default=None)
    parser.add_argument("--xpassthrough", type=str, required=False, default="false")
    args = parser.parse_args()
    data = process_results_data(args)
    # Below 10000
    print(data.describe())
    fig = plt.figure(figsize=(5, 5))
    build_plot(fig, data, args)
    plt.savefig(args.output_image_file, dpi=500)
