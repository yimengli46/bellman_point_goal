import lsp
import numpy as np
import matplotlib.pyplot as plt


def plot_pose(ax, pose, color='black', filled=True):
    if filled:
        ax.scatter(pose.x, pose.y, color=color, s=10, label='point')
    else:
        ax.scatter(pose.x,
                   pose.y,
                   color=color,
                   s=10,
                   label='point',
                   facecolors='none')


def plot_path(ax, path, style='b:'):
    if path is not None:
        ax.plot(path[0, :], path[1, :], style)


def plot_pose_path(ax, poses, style='b'):
    path = np.array([[p.x, p.y] for p in poses]).T
    plot_path(ax, path, style)


def plot_grid_with_frontiers(ax,
                             grid_map,
                             known_map,
                             frontiers,
                             cmap_name='viridis'):
    grid = lsp.utils.plotting.make_plotting_grid(grid_map, known_map)

    cmap = plt.get_cmap(cmap_name)
    for frontier in frontiers:
        color = cmap(frontier.prob_feasible)
        grid[frontier.points[0, :], frontier.points[1, :], 0] = color[0]
        grid[frontier.points[0, :], frontier.points[1, :], 1] = color[1]
        grid[frontier.points[0, :], frontier.points[1, :], 2] = color[2]

    ax.imshow(np.transpose(grid, (1, 0, 2)))
