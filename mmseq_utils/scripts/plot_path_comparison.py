import argparse
import os

import pandas
import numpy as np

from mmseq_utils.logging import DataLogger, DataPlotter, multipage, construct_logger
from mmseq_utils import math
import matplotlib.pyplot as plt

EE_POS_Target0 = np.array([1.5, -2.0])
EE_POS_Target1 = np.array([3.5, -1.0])     # home position + 4m in x direciton
Rc = 0.5 + 0.25         # robot radius + padding

def closet_approaching_pose(curr_base_pos, curr_ee_pos, next_ee_pos, rc):
    # line segment a that connecting curr_base_pos with next_ee_pos
    a = (next_ee_pos - curr_base_pos)
    a_hat = a / np.linalg.norm(a)

    n_hat = np.cross([0,0,1], np.hstack((a_hat, 0)))

    approach_point = curr_ee_pos + n_hat[:2] * rc
    approch_heading = np.arctan2(a[1], a[0])

    return np.hstack((approach_point, approch_heading))

def path_comparison(folder_path):
    plotters = []
    for filename in sorted(os.listdir(folder_path)):
        d = os.path.join(folder_path, filename)
        if os.path.isdir(d):
            plotter = construct_logger(d)
            plotters.append(plotter)

    num_plotters = len(plotters)
    axes = None
    for id, p in enumerate(plotters):
        axes = p.plot_base_path(axes, id, worldframe=False)    

    plotters[-1].plot_base_ref_path(axes, num_plotters, legend="", worldframe=False)
    # plotters[0].plot_ee_waypoints(axes, num_plotters, legend="")

    r_ew_w_ds = plotters[-1].data.get("r_ew_b_ds", [])
    r_bw_w_ds = plotters[-1].data.get("r_bw_b_ds", [])
    base_ref_path_text_pos = (r_bw_w_ds[0] + r_bw_w_ds[-1]) / 2
    base_ref_path_text_pos += np.array([0, 0.02])
    axes.text(base_ref_path_text_pos[0], base_ref_path_text_pos[1], "Base Reference Path", horizontalalignment='center')

    axes.scatter(EE_POS_Target0[0],EE_POS_Target0[1],  edgecolor='b', facecolor='b')
    axes.scatter(EE_POS_Target1[0],EE_POS_Target1[1], color='b')
    axes.text(EE_POS_Target0[0], EE_POS_Target0[1]-0.2, "EE Waypoint #1", horizontalalignment='center')
    axes.text(EE_POS_Target1[0], EE_POS_Target1[1]+0., "EE Waypoint #2", horizontalalignment='right')


    approach_pose = closet_approaching_pose(np.zeros(2), EE_POS_Target0[:2], EE_POS_Target1[:2], Rc)

    arrow_length = 0.2
    axes.arrow(approach_pose[0], approach_pose[1],
               arrow_length*np.cos(approach_pose[2]), arrow_length*np.sin(approach_pose[2]), width=0.01, facecolor='r', edgecolor='r')
    closet_approaching_pose_text_pos = approach_pose[:2] + np.array([np.cos(approach_pose[2]), np.sin(approach_pose[2])]) * arrow_length * 1.5
    axes.text(closet_approaching_pose_text_pos[0], closet_approaching_pose_text_pos[1], "Intermediate\nBase Waypoint")

    circle1 = plt.Circle(r_ew_w_ds[0][:2], Rc, facecolor=[1,0,0,0.25], edgecolor='r')
    axes.add_patch(circle1)

    circle_text_pos = EE_POS_Target0 + Rc * np.array([np.cos(2/3*np.pi), np.sin(2/3*np.pi) * 1.2])
    axes.text(circle_text_pos[0], circle_text_pos[1], "Closest\nApproaching Circle", horizontalalignment="center")

    axes.set_aspect("equal")
    plt.legend()
    plt.grid('on')



    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, help="Path to data folder.")
    parser.add_argument("--compare", action="store_true",
                        help="plot comparisons")
    args = parser.parse_args()

    path_comparison(args.folder)