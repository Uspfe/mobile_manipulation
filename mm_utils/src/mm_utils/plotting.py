"""
Plotting utilities for mobile manipulation experiments.

This module provides plotting functionality organized into logical sections:
- DataPlotter: Core data loading and processing
- Trajectory plotting: Path and tracking visualization
- Utility functions: PDF export, logger construction
"""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from spatialmath.base import rotz

import mm_control.MPC as MPC
from mm_utils import math, parsing
from mm_utils.math import wrap_pi_array

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def construct_logger(
    path_to_folder,
    process=True,
    data_plotter_class=None,
):
    """Load data from an experiment folder.

    Expected folder structure:
        <path_to_folder>/
            sim/
                data.npz
            control/
                data.npz
                config.yaml

    For combined sim+control runs (single process):
        <path_to_folder>/
            combined/
                data.npz
                config.yaml
    """
    if data_plotter_class is None:
        data_plotter_class = DataPlotter

    items = set(os.listdir(path_to_folder))

    if "sim" in items and "control" in items:
        # ROS nodes: separate sim/ and control/ folders
        return data_plotter_class.from_ROSSIM_results(path_to_folder, process)
    elif len(items) == 1:
        # Combined run: single subfolder (e.g., combined/)
        subfolder = list(items)[0]
        return data_plotter_class.from_PYSIM_results(
            os.path.join(path_to_folder, subfolder), process
        )
    elif "data.npz" in items and "config.yaml" in items:
        # Direct data files in folder
        return data_plotter_class.from_PYSIM_results(path_to_folder, process)
    else:
        raise ValueError(f"Unrecognized folder structure in {path_to_folder}.")


# =============================================================================
# MPC PLOTTING MIXIN
# =============================================================================


class MPCPlotterMixin:
    """Mixin class containing MPC-specific plotting methods."""

    def plot_cost(self):
        """Plot MPC cost function over time."""
        t_sim = self.data["ts"]
        cost_final = self.data.get("mpc_cost_finals")

        if cost_final is None:
            print("No cost data found")
            return

        f, ax = plt.subplots(1, 1)

        # Convert to numpy array and handle different shapes
        cost_final = np.array(cost_final)
        if cost_final.ndim == 1:
            ax.plot(
                t_sim,
                cost_final,
                ".-",
                label=self.data["name"],
                linewidth=2,
                markersize=8,
            )
        else:
            # Multi-dimensional cost - plot first dimension
            ax.plot(
                t_sim,
                cost_final[:, 0],
                ".-",
                label=self.data["name"],
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cost")
        ax.legend()
        ax.set_title("MPC Cost")
        ax.grid(True)

    def plot_run_time(self):
        """Plot controller execution time."""
        t_sim = self.data["ts"]
        run_time = self.data.get("controller_run_time")
        if run_time is None:
            print("Ignore run time")
            return

        f, ax = plt.subplots(1, 1)
        ax.plot(t_sim, run_time * 1000, label=self.data["name"], linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("run time (ms)")
        ax.legend()
        ax.set_title("Controller Run Time")


# =============================================================================
# TRAJECTORY PLOTTING MIXIN
# =============================================================================


class TrajectoryPlotterMixin:
    """Mixin class containing trajectory and path plotting methods."""

    def plot_ee_tracking(self):
        """Plot end-effector position tracking."""
        ts = self.data["ts"]
        r_ew_w_ds = self.data.get("r_ew_w_ds", [])
        r_ew_ws = self.data.get("r_ew_ws", [])

        if len(r_ew_w_ds) == 0 and len(r_ew_ws) == 0:
            return

        _, axes = plt.subplots(1, 1, sharex=True)
        legend = self.data["name"]

        if len(r_ew_w_ds) > 0:
            axes.plot(
                ts, r_ew_w_ds[:, 0], label=legend + "$x_d$", color="r", linestyle="--"
            )
            axes.plot(
                ts, r_ew_w_ds[:, 1], label=legend + "$y_d$", color="g", linestyle="--"
            )
            axes.plot(
                ts, r_ew_w_ds[:, 2], label=legend + "$z_d$", color="b", linestyle="--"
            )
        if len(r_ew_ws) > 0:
            axes.plot(ts, r_ew_ws[:, 0], label=legend + "$x$", color="r")
            axes.plot(ts, r_ew_ws[:, 1], label=legend + "$y$", color="g")
            axes.plot(ts, r_ew_ws[:, 2], label=legend + "$z$", color="b")
        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Position (m)")
        axes.set_title("End effector position tracking")

        return axes

    def plot_base_path(self):
        """Plot base path."""
        r_b = self.data.get("r_bw_ws", [])

        if len(r_b) == 0:
            return

        _, ax = plt.subplots(1, 1)

        if len(r_b) > 0:
            r_b = np.array(r_b)  # Convert to numpy array
            ax.plot(r_b[:, 0], r_b[:, 1], label=self.data["name"], linewidth=1)

        ax.grid()
        ax.legend()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Base Path Tracking")

    def plot_tracking_err(self):
        """Plot tracking error."""
        ts = self.data["ts"]

        _, ax = plt.subplots(1, 1)

        ax.plot(
            ts,
            self.data["err_base"],
            label=self.data["name"]
            + f" base err, rms={self.data['statistics']['err_base']['rms']:.3f}",
            linestyle="--",
        )
        ax.plot(
            ts,
            self.data["err_ee"],
            label=self.data["name"]
            + f" EE err, rms={self.data['statistics']['err_ee']['rms']:.3f}",
            linestyle="-",
        )

        ax.grid()
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (m)")
        ax.set_title("Tracking Error vs Time")

    def plot_task_performance(self):
        """Plot task performance metrics."""
        f, axes = plt.subplots(4, 1, sharex=True)

        legend = self.data["name"]
        t_sim = self.data["ts"]

        axes[0].plot(
            t_sim,
            self.data["constraints_violation"] * 100,
            label=f"{legend} mean={self.data['statistics']['constraints_violation']['mean']*100:.1f}%",
        )
        axes[0].set_ylabel("Constraints violation (%)")

        axes[1].plot(
            t_sim,
            self.data["err_ee"],
            label=f"{legend} acc={self.data['statistics']['err_ee']['integral']:.3f}",
        )
        axes[1].set_ylabel("EE Error (m)")

        axes[2].plot(
            t_sim,
            self.data["err_base"],
            label=f"{legend} acc={self.data['statistics']['err_base']['integral']:.3f}",
        )
        axes[2].set_ylabel("Base Error (m)")

        axes[3].plot(t_sim, self.data["arm_manipulability"], label=legend)
        axes[3].set_ylabel("Arm Manipulability")
        axes[3].set_xlabel("Time (s)")

        for ax in axes:
            ax.legend()
            ax.grid(True)


# =============================================================================
# CORE DATAPLOTTER CLASS
# =============================================================================


class DataPlotter(TrajectoryPlotterMixin, MPCPlotterMixin):
    """Core class for loading and processing simulation data for plotting."""

    def __init__(self, data, config=None, process=True):
        self.data = data
        self.data["name"] = self.data.get("name", "data")
        self.name = self.data["name"]
        self.config = config
        if config is not None:
            # controller
            control_class = getattr(MPC, config["controller"]["type"], None)
            if control_class is None:
                raise ValueError(
                    f"Unknown controller type: {config['controller']['type']}"
                )

            config["controller"]["acados"]["cython"]["enabled"] = True
            config["controller"]["acados"]["cython"]["recompile"] = False
            self.controller = control_class(config["controller"])
            self.model_interface = self.controller.model_interface

        if process:
            self._post_processing()
            self._get_statistics()

    @classmethod
    def from_logger(cls, logger, process):
        # convert logger data to numpy format
        data = {}
        for key, value in logger.data.items():
            data[key] = np.array(value)
        return cls(data, process=process)

    @classmethod
    def from_npz(cls, npz_file_path, process):
        data = dict(np.load(npz_file_path))
        if "name" not in data:
            path_split = npz_file_path.split("/")
            folder_name = path_split[-2]
            data_name = folder_name.split("_")[:-2]
            data_name = "_".join(data_name)
            data["name"] = data_name
        return cls(data, process=process)

    @classmethod
    def from_PYSIM_results(cls, folder_path, process):
        """For data obtained from running controller in the simulation loop"""
        npz_file_path = os.path.join(folder_path, "data.npz")
        data = dict(np.load(npz_file_path, allow_pickle=True))
        config_file_path = os.path.join(folder_path, "config.yaml")
        config = parsing.load_config(config_file_path)
        folder_name = folder_path.split("/")[-1]
        data["name"] = folder_name.split("_")[0]
        data["folder_path"] = folder_path

        return cls(data, config, process=process)

    @classmethod
    def from_ROSSIM_results(cls, folder_path, process):
        """For data from running simulation and controller as two ROS nodes."""
        data_decoupled = {}
        config = None

        sim_path = os.path.join(folder_path, "sim", "data.npz")
        control_path = os.path.join(folder_path, "control", "data.npz")
        config_path = os.path.join(folder_path, "control", "config.yaml")

        data_decoupled["sim"] = dict(np.load(sim_path, allow_pickle=True))
        data_decoupled["control"] = dict(np.load(control_path, allow_pickle=True))
        config = parsing.load_config(config_path)

        data = data_decoupled["control"]

        t = data["ts"]
        t_sim = data_decoupled["sim"]["ts"]
        for key, value in data_decoupled["sim"].items():
            if key in [
                "ts",
                "sim_timestep",
                "nq",
                "nv",
                "nx",
                "nu",
                "duration",
                "dir_path",
                "cmd_vels",
            ]:
                continue
            else:
                value = np.array(value)
                f_interp = interp1d(t_sim, value, axis=0, fill_value="extrapolate")
                data[key] = f_interp(t)
        data["ts"] -= data["ts"][0]
        data["name"] = folder_path.split("/")[-1]
        data["folder_path"] = folder_path

        return cls(data, config, process)

    def _get_tracking_err(self, ref_name, robot_traj_name):
        N = len(self.data["ts"])
        rs = self.data.get(robot_traj_name, None)
        rds = self.data.get(ref_name, None)

        if rds is None:
            return np.zeros(N)
        if rs is None:
            rs = np.zeros_like(rds)

        if len(rs) == len(rds):
            # Handle dimension mismatch - for EE tracking, compare only position (first 3 elements)
            if rs.shape[1] != rds.shape[1]:
                # Take minimum dimensions to compare (typically position only)
                min_dim = min(rs.shape[1], rds.shape[1])
                rs_pos = rs[:, :min_dim]
                rds_pos = rds[:, :min_dim]
                errs = np.linalg.norm(rds_pos - rs_pos, axis=1)
            else:
                errs = np.linalg.norm(rds - rs, axis=1)
        else:
            errs = np.zeros(len(rs))
        return errs

    def _transform_w2b_SE3(self, qb, r_w):
        Rbw = rotz(-qb[2])
        rbw = np.array([qb[0], qb[1], 0])
        r_b = (Rbw @ (r_w - rbw).T).T

        return r_b

    def _transform_w2b_SE2(self, qb, r_w):
        Rbw = rotz(-qb[2])[:2, :2]
        rbw = np.array(qb[:2])
        r_b = (Rbw @ (r_w - rbw).T).T

        return r_b

    def _get_mean_violation(self, data_normalized):
        vio_mask = data_normalized > 1.05
        vio = np.sum((data_normalized - 1) * vio_mask, axis=1)
        vio_num = np.sum(vio_mask, axis=1)
        vio_mean = np.where(vio_num > 0, vio / vio_num, 0)
        return vio_mean, np.sum(vio_num)

    def _post_processing(self):
        # tracking error
        self.data["err_ee"] = self._get_tracking_err("r_ew_w_ds", "r_ew_ws")
        self.data["err_base"] = self._get_tracking_err("r_bw_w_ds", "r_bw_ws")
        self.data["err_ee_normalized"] = self.data["err_ee"] / self.data["err_ee"][0]
        self.data["err_base_normalized"] = (
            self.data["err_base"] / self.data["err_base"][0]
        )

        # signed distance
        nq = self.data["nq"]
        qs = self.data["xs"][:, :nq]

        print(self.data["xs"].shape)

        # keyed by obstacle names or "self"
        names = ["self", "static_obstacles"]
        params = {"self": [], "static_obstacles": []}
        if self.config["controller"]["sdf_collision_avoidance_enabled"]:
            names += ["sdf"]
            sdf_param_names = [
                "_".join(["mpc", "sdf", "param", str(i)]) + "s"
                for i in range(self.model_interface.sdf_map.dim + 1)
            ]
            sdf_param = [self.data[name] for name in sdf_param_names]
            params["sdf"] = sdf_param
        sds_dict = self.model_interface.evaluteSignedDistance(
            names, qs, copy.deepcopy(params)
        )
        sds = np.array([sd for sd in sds_dict.values()])
        for id, name in enumerate(names):
            self.data["_".join(["signed_distance", name])] = sds_dict[name]
        self.data["signed_distance"] = np.min(sds, axis=0)

        names = []
        params = {}

        if self.config["controller"]["self_collision_avoidance_enabled"]:
            names += ["self"]
            params = {"self": []}

        if self.config["controller"]["sdf_collision_avoidance_enabled"]:
            param_names = [
                "_".join(["mpc", "sdf", "param", str(i)]) + "s"
                for i in range(self.model_interface.sdf_map.dim + 1)
            ]
            sdf_params = [self.data[name] for name in param_names]
            params["sdf"] = sdf_params
            names += ["sdf"]

        if self.config["controller"]["static_obstacles_collision_avoidance_enabled"]:
            params["static_obstacles"] = []
            names += ["static_obstacles"]
        sds_dict_detailed = self.model_interface.evaluteSignedDistancePerPair(
            names, qs, params
        )
        self.data["signed_distance_detailed"] = {}

        for name, sds in sds_dict_detailed.items():
            self.data["signed_distance_detailed"][name] = {}
            for pair, sd in sds.items():
                self.data["signed_distance_detailed"][name][pair] = sd

        # normalized state and input w.r.t bounds
        # -1 --> saturate lower bounds
        # 1  --> saturate upper bounds
        # 0  --> in middle
        bounds = self.config["controller"]["robot"]["limits"]
        self.data["xs_normalized"] = math.normalize_wrt_bounds(
            parsing.parse_array(bounds["state"]["lower"]),
            parsing.parse_array(bounds["state"]["upper"]),
            self.data["xs"],
        )
        self.data["cmd_vels_normalized"] = math.normalize_wrt_bounds(
            parsing.parse_array(bounds["state"]["lower"])[nq:],
            parsing.parse_array(bounds["state"]["upper"])[nq:],
            self.data["cmd_vels"],
        )
        self.data["cmd_accs_normalized"] = math.normalize_wrt_bounds(
            parsing.parse_array(bounds["input"]["lower"]),
            parsing.parse_array(bounds["input"]["upper"]),
            self.data["cmd_accs"],
        )

        # box constraints
        constraints_violation = np.abs(
            np.hstack((self.data["xs_normalized"], self.data["cmd_accs_normalized"]))
        )
        constraints_violation = np.hstack(
            (
                constraints_violation,
                np.expand_dims(0.05 - self.data["signed_distance"], axis=1) / 0.05 + 1,
            )
        )
        (
            self.data["constraints_violation"],
            self.data["constraints_violation_num"],
        ) = self._get_mean_violation(constraints_violation)

        # singularity
        man_fcn = self.model_interface.robot.manipulability_fcn
        man_fcn_map = man_fcn.map(len(self.data["ts"]))
        manipulability = man_fcn_map(self.data["xs"][:, :nq].T)
        self.data["manipulability"] = manipulability.toarray().flatten()

        arm_man_fcn = self.model_interface.robot.arm_manipulability_fcn
        arm_man_fcn_map = arm_man_fcn.map(len(self.data["ts"]))
        arm_manipulability = arm_man_fcn_map(self.data["xs"][:, :nq].T)
        self.data["arm_manipulability"] = arm_manipulability.toarray().flatten()

        # jerk
        self.data["cmd_jerks"] = (
            self.data["cmd_accs"][1:, :] - self.data["cmd_accs"][:-1, :]
        ) / np.expand_dims(self.data["ts"][1:] - self.data["ts"][:-1], axis=1)

        # coordinate transform
        qb = self.data["xs"][0, :3]

        # Transform only position components (first 3 dimensions)
        self.data["r_ew_bs"] = self._transform_w2b_SE3(qb, self.data["r_ew_ws"][:, :3])
        if "r_ew_w_ds" in self.data.keys():
            self.data["r_ew_b_ds"] = self._transform_w2b_SE3(
                qb, self.data["r_ew_w_ds"][:, :3]
            )

        # has_rb = self.data.get("r_bw_w_ds", None)
        self.data["r_bw_bs"] = self._transform_w2b_SE2(qb, self.data["r_bw_ws"])
        if "r_bw_w_ds" in self.data.keys():
            self.data["r_bw_b_ds"] = self._transform_w2b_SE2(qb, self.data["r_bw_w_ds"])
        if "yaw_bw_w_ds" in self.data.keys():
            self.data["yaw_bw_w_ds"] -= qb[2]
            self.data["yaw_bw_w_ds"] = wrap_pi_array(self.data["yaw_bw_w_ds"])
        if "yaw_bw_ws" in self.data.keys():
            self.data["yaw_bw_ws"] -= qb[2]
            self.data["yaw_bw_ws"] = wrap_pi_array(self.data["yaw_bw_ws"])

        N = len(self.data["ts"])

        self.data["mpc_ee_predictions"] = []
        self.data["mpc_base_predictions"] = []

        print(self.data["mpc_x_bars"].shape)
        for t_index in range(N):
            x_bar = self.data["mpc_x_bars"][t_index]
            ee_bar, base_bar = self.controller._getEEBaseTrajectories(x_bar)
            self.data["mpc_ee_predictions"].append(ee_bar)
            self.data["mpc_base_predictions"].append(base_bar)

        self.data["mpc_ee_predictions"] = np.array(self.data["mpc_ee_predictions"])
        self.data["mpc_base_predictions"] = np.array(self.data["mpc_base_predictions"])

    def _get_statistics(self):
        self.data["statistics"] = {}
        # EE tracking error
        err_ee_stats = math.statistics(self.data["err_ee"])
        self.data["statistics"]["err_ee"] = {
            "rms": math.rms_continuous(self.data["ts"], self.data["err_ee"]),
            "integral": math.integrate_zoh(self.data["ts"], self.data["err_ee"]),
            "mean": err_ee_stats[0],
            "max": err_ee_stats[1],
            "min": err_ee_stats[2],
            "std": math.statistics_std(self.data["err_ee"]),
        }
        # base tracking error
        err_base_stats = math.statistics(self.data["err_base"])
        self.data["statistics"]["err_base"] = {
            "rms": math.rms_continuous(self.data["ts"], self.data["err_base"]),
            "integral": math.integrate_zoh(self.data["ts"], self.data["err_base"]),
            "mean": err_base_stats[0],
            "max": err_base_stats[1],
            "min": err_base_stats[2],
        }

        # EE tracking error (Normalized)
        err_ee_normalized_stats = math.statistics(self.data["err_ee_normalized"])
        self.data["statistics"]["err_ee_normalized"] = {
            "rms": math.rms_continuous(self.data["ts"], self.data["err_ee_normalized"]),
            "integral": math.integrate_zoh(
                self.data["ts"], self.data["err_ee_normalized"]
            ),
            "mean": err_ee_normalized_stats[0],
            "max": err_ee_normalized_stats[1],
            "min": err_ee_normalized_stats[2],
        }
        # base tracking error (Normalized)
        err_base_normalized_stats = math.statistics(self.data["err_base_normalized"])
        self.data["statistics"]["err_base_normalized"] = {
            "rms": math.rms_continuous(
                self.data["ts"], self.data["err_base_normalized"]
            ),
            "integral": math.integrate_zoh(
                self.data["ts"], self.data["err_base_normalized"]
            ),
            "mean": err_base_normalized_stats[0],
            "max": err_base_normalized_stats[1],
            "min": err_base_normalized_stats[2],
        }

        # signed distance
        sd_stats = math.statistics(self.data["signed_distance"])
        self.data["statistics"]["signed_distance"] = {
            "mean": sd_stats[0],
            "max": sd_stats[1],
            "min": sd_stats[2],
        }

        # bounds saturation
        nq = self.data["nq"]
        q_stats = math.statistics(np.abs(self.data["xs_normalized"][:, :nq].flatten()))
        self.data["statistics"]["q_saturation"] = {
            "mean": q_stats[0],
            "max": q_stats[1],
            "min": q_stats[2],
        }

        qdot_stats = math.statistics(
            np.abs(self.data["xs_normalized"][:, nq:].flatten())
        )
        self.data["statistics"]["qdot_saturation"] = {
            "mean": qdot_stats[0],
            "max": qdot_stats[1],
            "min": qdot_stats[2],
        }

        cmd_vels_stats = math.statistics(
            np.abs(self.data["cmd_vels_normalized"].flatten())
        )
        self.data["statistics"]["cmd_vels_saturation"] = {
            "mean": cmd_vels_stats[0],
            "max": cmd_vels_stats[1],
            "min": cmd_vels_stats[2],
        }

        cmd_accs_stats = math.statistics(
            np.abs(self.data["cmd_accs_normalized"].flatten())
        )
        self.data["statistics"]["cmd_accs_saturation"] = {
            "mean": cmd_accs_stats[0],
            "max": cmd_accs_stats[1],
            "min": cmd_accs_stats[2],
        }

        cmd_jerks_base_linear_stats = math.statistics(
            np.linalg.norm(self.data["cmd_jerks"][:, :2], axis=1).flatten()
        )
        self.data["statistics"]["cmd_jerks_base_linear"] = {
            "mean": cmd_jerks_base_linear_stats[0],
            "max": cmd_jerks_base_linear_stats[1],
            "min": cmd_jerks_base_linear_stats[2],
        }
        cmd_jerks_base_angular_stats = math.statistics(
            np.abs(self.data["cmd_jerks"][:, 2])
        )
        self.data["statistics"]["cmd_jerks_base_angular"] = {
            "mean": cmd_jerks_base_angular_stats[0],
            "max": cmd_jerks_base_angular_stats[1],
            "min": cmd_jerks_base_angular_stats[2],
        }

        cmd_jerks_stats = math.statistics(self.data["cmd_jerks"])
        self.data["statistics"]["cmd_jerks"] = {
            "mean": cmd_jerks_stats[0],
            "max": cmd_jerks_stats[1],
            "min": cmd_jerks_stats[2],
        }

        violation_stats = math.statistics(self.data["constraints_violation"])
        self.data["statistics"]["constraints_violation"] = {
            "mean": violation_stats[0],
            "max": violation_stats[1],
            "min": violation_stats[2],
            "num": self.data["constraints_violation_num"],
        }
        run_time_states = math.statistics(self.data["controller_run_time"])
        self.data["statistics"]["run_time"] = {
            "mean": run_time_states[0],
            "max": run_time_states[1],
            "min": run_time_states[2],
        }
        print(self.data["statistics"]["constraints_violation"]["num"])

    def summary(self, stat_names):
        """get a summary of statistics

        :param stat_names: list of stats of interests, (key, value) pairs
        :return: array
        """

        stats = []
        stats_dict = self.data["statistics"]

        # Return None if either key or val doesn't exist
        for key, val in stat_names:
            stats.append(stats_dict.get(key, {}).get(val, None))

        return stats

    # Convenience methods for common plotting workflows
    def plot_all(self):
        """Plot all available data."""
        self.plot_tracking()
        self.plot_mpc()

    def plot_tracking(self):
        """Plot tracking performance."""
        self.plot_ee_tracking()
        self.plot_base_path()
        self.plot_tracking_err()
        self.plot_task_performance()
        plt.show()

    def plot_mpc(self):
        """Plot MPC-specific data."""
        self.plot_cost()
        self.plot_run_time()
        plt.show()
