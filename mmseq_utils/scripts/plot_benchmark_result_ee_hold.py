import argparse
import os
from typing import Optional, List, Dict, Tuple, Union

import pandas
import numpy as np

from mmseq_utils.plotting import construct_logger, DataPlotter, HTMPCDataPlotter, ROSBagPlotter
from mmseq_utils import parsing
from mmseq_utils.matplotlib_helper import plot_square_box, plot_circle, plot_cross
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
STYLE_PATH = parsing.parse_ros_path({"package": "mmseq_utils", "path":"scripts/plot_style.mplstyle"})
plt.style.use(STYLE_PATH)
plt.rcParams['pdf.fonttype'] = 42


BOX_CENTER = [3.6, -0.3, 0.3]
BASE_RADIUS = 0.55
YELLOW="#FFAE42"
SCARLET="#cc1d00"
EE_WAYPOINT = [3.6, -0.3, 0.85]
BASE_WAYPOINTS = np.array([[0, 0, 0],
                            [2.5, -0.0, 0.0],
                            [7, 1.0, 0.0]]) 
TASK_CHANGE_TIME = 1729790633.381

class EEHoldDataPlotter(ROSBagPlotter):
    def __init__(self, folder_path):
        config = None

        for filename in os.listdir(folder_path):
            d = os.path.join(folder_path, filename)

            if os.path.isdir(d):
                path_to_control_folder = d
            else:
                path_to_bag = d

        path_to_config = os.path.join(path_to_control_folder, "config.yaml")

        super().__init__(path_to_bag, path_to_config)
        self.data["name"] = folder_path.split('/')[-1]

        self._trim_data()
    
    def _trim_data(self):
        # find time start when base position reset to 0
        r_b = self.data["ridgeback"]["joint_states"]["qs"]

        time_start_index_base = np.where(np.linalg.norm(r_b[:, :2], axis=-1) < 0.01)[0][0]
        time_end_index_base = np.where(self.data["ridgeback"]["joint_states"]["ts"] < TASK_CHANGE_TIME)[0][-1]

        print("Trim data starting from {}".format(time_start_index_base))
        time_start = self.data["ridgeback"]["joint_states"]["ts"][time_start_index_base]
        
        # trim base
        for key, data in self.data["ridgeback"]["joint_states"].items():
            self.data["ridgeback"]["joint_states"][key] = data[time_start_index_base:time_end_index_base]
        
        # trime ee
        time_start_index_ee = np.where(self.data["model"]["EE"]['ts'] - time_start < 0.01)[0][-1]
        time_end_index_ee = np.where(self.data["model"]["EE"]['ts'] < TASK_CHANGE_TIME)[0][-1]

        print("Trim EE data starting from {}".format(time_start_index_ee))

        for key, data in self.data["model"]["EE"].items():
            self.data["model"]["EE"][key] = data[time_start_index_ee:time_end_index_ee]

    def plot_base_path(self, axes=None, index=0, legend=None, worldframe=True, linewidth=1, color=None, linestyle="-"):
        r_b = self.data["ridgeback"]["joint_states"]["qs"]

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        # if legend == "baseline h = 0.8m":
        #     r_ew_ws += [0, 1]
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        if color is None:
            color = colors[index]
        if len(r_b) > 0:
            axes.plot(r_b[:, 0], r_b[:, 1], label=legend, color=color, linewidth=linewidth, linestyle=linestyle)

        axes.grid()
        axes.legend()
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        axes.set_title("Base Path")

        return axes

    def plot_ee_path(self, axes=None, index=0, legend=None, worldframe=True, linewidth=1, color=None, linestyle="-"):

        r_ee = self.data["model"]["EE"]['pos']

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        # if legend == "baseline h = 0.8m":
        #     r_ew_ws += [0, 1]
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        if color is None:
            color = colors[index]


        if len(r_ee) > 0:
            axes.plot(r_ee[:, 0], r_ee[:, 1], label=legend, color=color, linewidth=linewidth, linestyle=linestyle)

        axes.grid()
        axes.legend()
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        axes.set_title("EE Path")

        return axes    

class EEHoldBenchmarkPlotter():
    plotters: List[EEHoldDataPlotter]

    def __init__(self, args) -> None:
        folder_path = args.folder
        self.plotters = []
        for filename in sorted(os.listdir(folder_path), reverse=True):
            d = os.path.join(folder_path, filename)
            if os.path.isdir(d):
                plotter = EEHoldDataPlotter(d)

                self.plotters.append(plotter)

        self.num_plotters = len(self.plotters)
        self.folder_path = folder_path

    def plot_base_and_ee_path(self):

        f, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5, 2.3))
        cm_index = [0.8, 0.6, 0.4, 0.2]
        line_width = [1.5, 2.0, 2.0, 2.0, 2.0]
        colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
        cm = LinearSegmentedColormap.from_list(
            "my_list", colors, N=50)
        
        # plot base path
        for ax in axes:
            plot_square_box(0.6, BOX_CENTER[:2], ax)

        for id, p in enumerate(self.plotters):
            axes[0] = p.plot_base_path(axes[0], id, worldframe=False, linewidth=line_width[id])
        axes[0].plot(BASE_WAYPOINTS[:, 0], BASE_WAYPOINTS[:, 1], '-.', markersize=10, color=SCARLET, label="Base Ref")

        for id, p in enumerate(self.plotters):
            axes[1] = p.plot_ee_path(axes[1], id, worldframe=False, linewidth=line_width[id])
        axes[1].plot(EE_WAYPOINT[0], EE_WAYPOINT[1], '.', markersize=10, color=SCARLET, label="EE Waypoint")

        # axes.set_aspect("equal")
        for ax in axes:
            ax.set_aspect('equal', 'box')
            ax.grid('on')
            ax.legend(loc="lower right")
            ax.set_xlim([0, 7])
        
        y_lim = axes[0].get_ylim()
        axes[1].set_ylim(y_lim)
        axes[0].xaxis.label.set_visible(False)


        xs = [2, 3.27]
        ys=  [-0.45, 0.5]
        for id, x in enumerate(xs):
            arrow_length = 1.5
            y = ys[id]
            axes[id].annotate(
                '', 
                xy=(x + arrow_length, y), 
                xytext=(x+arrow_length*0.05, y),
                arrowprops=dict(arrowstyle='<->', linewidth=2)
            )

            # Plot left-pointing arrow
            axes[id].annotate(
                '', 
                xy=(x - arrow_length, y), 
                xytext=(x-arrow_length*0.05, y),
                arrowprops=dict(arrowstyle='<->', linewidth=2)
            )

            # Add text annotation at the center
            axes[id].text(
                x+arrow_length/2, y, "Man",
                ha='center', va='center', 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
            )
            axes[id].text(
                x-arrow_length/2, y, "Nav",
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
            )
            axes[id].axvspan(0, x, color="red", alpha=0.1)  # alpha for transparency
            axes[id].axvspan(x, 7, color="green", alpha=0.1)  # alpha for transparency
            axes[id].axvline(x, color='darkgreen', linestyle='--')  # Optional visual for x-value


        plt.subplots_adjust(top=0.952,
                            bottom=0.12,
                            left=0.152,
                            right=0.967,
                            hspace=0.045,
                            wspace=0.2)



        f = plt.gcf()
        f.savefig(self.folder_path + "/ee_base_path.pdf" , pad_inches=0)

    def plot_model_vs_groundtruth(self):

        for p in self.plotters:
            p.plot_model_vs_groundtruth()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, help="Path to data folder.")


    args = parser.parse_args()

    plotter = EEHoldBenchmarkPlotter(args)
    plotter.plot_base_and_ee_path()
    plotter.plot_model_vs_groundtruth()

    plt.show()

