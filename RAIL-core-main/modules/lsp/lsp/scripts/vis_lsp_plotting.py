import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import re
import scipy.stats


def process_results_data(args):
    # Load data and loop through rows
    data = []
    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . learned: (.*?) . baseline: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1]), float(d[2])])

    return pd.DataFrame(data, columns=['seed', 'learned_cost', 'baseline_cost'])


def build_plot(fig, data, args, cmap='Blues'):
    xy = np.vstack([data['baseline_cost'], data['learned_cost']])
    z = scipy.stats.gaussian_kde(xy)(xy)

    data['zs'] = z
    data = data.sort_values(by=['zs'])
    z = data['zs']
    colors = cm.get_cmap(cmap)((z - z.min()) / (z.max() - z.min()) * 0.75 + 0.25)

    fig.gca()
    ax = plt.subplot(111)
    ax.scatter(data['baseline_cost'], data['learned_cost'], c=colors)
    ax.set_aspect('equal', adjustable='box')
    cb = min(max(data['baseline_cost']), max(data['learned_cost']))
    ax.plot([0, cb], [0, cb], 'k')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a figure (and write to file) for results from the interpretability project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file',
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument('--output_image_file',
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument('--xpassthrough',
                        type=str,
                        required=False,
                        default='false')
    args = parser.parse_args()

    data = process_results_data(args)

    # Below 10000
    print(data.describe())
    fig = plt.figure(dpi=300, figsize=(5, 5))
    build_plot(fig, data, args)
    plt.savefig(args.output_image_file, dpi=300)
