import json
import numpy as np
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_single_aruco_board(board_config):
    with open(board_config, "r") as f:
        board_data = json.load(f)

    # load ids and corners from config
    marker_ids_list = []
    marker_corners_list = []

    for marker in board_data["markers"]:
        marker_ids_list.append(marker["id"])
        marker_corners_list.append(marker["corners"])

    return marker_ids_list, marker_corners_list

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_squares(ids, squares):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o', 's', '^', 'd']  # Different marker symbols
    colors = plt.cm.viridis(np.linspace(0, 1, len(squares)))  # Generate unique colors
    
    for key, vertices, color in zip(ids, squares, colors):
        if len(vertices) == 4:
            square = Poly3DCollection([vertices], alpha=0.5, edgecolor='k', facecolor=color)
            ax.add_collection3d(square)
            for idx, vertex in enumerate(vertices):
                ax.scatter(vertex[0], vertex[1], vertex[2], marker=markers[idx % len(markers)], color=color, label=f'ID {key} P{idx+1}')
    
    set_axes_equal(ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("filepath", nargs="?", default="default", help="filepath")
    args = parser.parse_args()

    ids, corners = load_single_aruco_board(args.filepath)
    plot_squares(ids, corners)