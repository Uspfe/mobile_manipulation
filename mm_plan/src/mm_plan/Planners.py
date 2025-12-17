"""Generic planners that work for both base and end-effector."""

import logging
from abc import ABC, abstractmethod

import numpy as np

from mm_utils.enums import PlannerType, RefDataType, RefType
from mm_utils.math import interpolate, wrap_pi_array, wrap_pi_scalar


class Planner(ABC):
    """Base class for planners"""

    def __init__(self, name, type, ref_type, ref_data_type):
        self.py_logger = logging.getLogger("Planner")
        self.name = name
        # The following variables are for automatically
        # (1) publishing rviz visualization data
        # (2) assigning the correct mpc cost function
        self.type = type
        self.ref_type = ref_type
        self.ref_data_type = ref_data_type
        self.robot_states = None
        self.close_to_finish = False

    @abstractmethod
    def getTrackingPoint(self, t, robot_states=None):
        """get tracking point for controllers

        :param t: time (s)
        :type t: float
        :param robot_states: (joint angle, joint velocity), defaults to None
        :type robot_states: tuple, optional

        :return: position, velocity
        :rtype: numpy array, numpy array
        """
        p = None
        v = None
        return p, v

    @abstractmethod
    def checkFinished(self, t, curr_pose, curr_vel=None):
        """check if the planner is finished

        :param t: time since the controller started
        :type t: float
        :param curr_pose: Current pose array in world frame
            - For BASE: [x, y, yaw]
            - For EE: [x, y, z, roll, pitch, yaw]
        :type curr_pose: numpy array
        :param curr_vel: Current velocity array (optional, for end_stop checks)
        :type curr_vel: numpy array, optional
        :return: true if the planner has finished, false otherwise
        :rtype: boolean
        """
        finished = True
        return finished

    def updateRobotStates(self, robot_states):
        """update robot states

        :param robot_states: (joint angle, joint velocity)
        :type robot_states: tuple
        """
        self.robot_states = robot_states

    def ready(self):
        """
        :return: true if the planner is ready to be called getTrackingPoint
        :rtype: boolean
        """
        return True

    def closeToFinish(self):
        """
        :return: true if the planner is close to finish
        :rtype: boolean
        """
        return self.close_to_finish

    def activate(self):
        self.started = True
        return


class TrajectoryPlanner(Planner):
    """Abstract base class for trajectory planners that interpolate along a trajectory.

    Subclasses must implement getTrackingPoint and checkFinished.
    """

    def __init__(self, name, type, ref_type, ref_data_type):
        super().__init__(name, type, ref_type, ref_data_type)

    def _interpolate(self, t, plan):
        """Interpolate along a trajectory plan."""
        p, v = interpolate(t, plan)
        return p, v

    def ready(self):
        """
        :return: true if the planner is ready to be called getTrackingPoint
        :rtype: boolean
        """
        return True

    @abstractmethod
    def getTrackingPointArray(self, robot_states, num_pts, dt, time_offset=0):
        """Get array of tracking points along the path.

        Must be implemented by subclasses.
        """
        pass


class WaypointPlanner(Planner):
    """Generic waypoint planner for base (SE2) and EE (SE3).

    All target_pose values are in world frame:
    - For base: target_pose is [x, y, yaw] in world frame
    - For EE: target_pose is [x, y, z, roll, pitch, yaw] in world frame
    """

    def __init__(self, config):
        frame_id_str = config.get("frame_id", "base")
        planner_type = PlannerType(frame_id_str)

        # Determine ref_data_type based on planner type
        if planner_type == PlannerType.BASE:
            ref_data_type = RefDataType.SE2
        else:  # EE
            ref_data_type = RefDataType.SE3

        super().__init__(
            name=config["name"],
            type=planner_type,
            ref_type=RefType.WAYPOINT,
            ref_data_type=ref_data_type,
        )

        self.target_pose = np.array(config["target_pose"])
        self.tracking_err_tol = config.get("tracking_err_tol", 0.02)
        self.tracking_ori_err_tol = config.get("tracking_ori_err_tol", 0.1)
        self.hold_period = config.get("hold_period", 0.0)  # Optional hold period for EE

        self.finished = False
        self.reached_target = False
        self.t_reached_target = 0

        # Validate dimensions
        if planner_type == PlannerType.BASE and len(self.target_pose) != 3:
            raise ValueError(
                f"Base waypoint must be SE2 [x, y, yaw], got {len(self.target_pose)} dimensions"
            )
        elif planner_type == PlannerType.EE and len(self.target_pose) != 6:
            raise ValueError(
                f"EE waypoint must be SE3 [x, y, z, roll, pitch, yaw], got {len(self.target_pose)} dimensions"
            )

    def getTrackingPoint(self, t, robot_states=None):
        """Return target pose and zero velocity."""
        return self.target_pose.copy(), np.zeros_like(self.target_pose)

    def checkFinished(self, t, curr_pose, curr_vel=None):
        """Check if waypoint has been reached.

        Args:
            t: Current time
            curr_pose: Current pose array in world frame
                - For BASE: [x, y, yaw]
                - For EE: [x, y, z, roll, pitch, yaw]
            curr_vel: Current velocity array (unused, kept for interface compatibility)
        """
        # For base SE2, compute proper rotation error
        if self.type == PlannerType.BASE:
            pos_err = np.linalg.norm(curr_pose[:2] - self.target_pose[:2])
            yaw_err = abs(wrap_pi_scalar(curr_pose[2] - self.target_pose[2]))
            # Check position and orientation separately
            pos_within_tol = pos_err < self.tracking_err_tol
            ori_within_tol = yaw_err < self.tracking_ori_err_tol
        else:  # EE SE3
            pos_err = np.linalg.norm(curr_pose[:3] - self.target_pose[:3])
            # Wrap Euler angles properly before computing orientation error
            ori_diff = wrap_pi_array(curr_pose[3:] - self.target_pose[3:])
            ori_err = np.linalg.norm(ori_diff)
            # Check position and orientation separately
            pos_within_tol = pos_err < self.tracking_err_tol
            ori_within_tol = ori_err < self.tracking_ori_err_tol

        if pos_within_tol and ori_within_tol:
            if not self.reached_target:
                self.reached_target = True
                self.t_reached_target = t
                if self.type == PlannerType.BASE:
                    self.py_logger.info(
                        f"{self.name} reached target (pos_err: {pos_err:.4f}, ori_err: {yaw_err:.4f})"
                    )
                else:
                    self.py_logger.info(
                        f"{self.name} reached target (pos_err: {pos_err:.4f}, ori_err: {ori_err:.4f})"
                    )

            # Check hold period (mainly for EE)
            if self.hold_period > 0 and (t - self.t_reached_target) < self.hold_period:
                return False

            if not self.finished:
                self.finished = True
                self.py_logger.info(f"{self.name} finished")
        else:
            # Reset if moved away from target
            if self.reached_target:
                self.reached_target = False
                self.t_reached_target = 0

        return self.finished

    def reset(self):
        """Reset planner state."""
        self.finished = False
        self.reached_target = False
        self.t_reached_target = 0
        self.py_logger.info(f"{self.name} reset")


class PathPlanner(TrajectoryPlanner):
    """Generic path planner for base (SE2) and EE (SE3).

    Uses pre-computed paths provided as arrays in the configuration.
    All path points are in world frame:
    - For base: path points are [x, y, yaw] in world frame
    - For EE: path points are [x, y, z, roll, pitch, yaw] in world frame
    """

    def __init__(self, config):
        frame_id_str = config.get("frame_id", "base")
        planner_type = PlannerType(frame_id_str)

        # Determine ref_data_type based on planner type
        if planner_type == PlannerType.BASE:
            ref_data_type = RefDataType.SE2
        else:  # EE
            ref_data_type = RefDataType.SE3

        super().__init__(
            name=config["name"],
            type=planner_type,
            ref_type=RefType.PATH,
            ref_data_type=ref_data_type,
        )

        self.tracking_err_tol = config.get("tracking_err_tol", 0.02)
        self.tracking_ori_err_tol = config.get("tracking_ori_err_tol", 0.1)
        self.end_stop = config.get("end_stop", False)

        self.finished = False
        self.started = False
        self.start_time = 0

        # Load pre-computed path
        if "path" not in config:
            raise ValueError("Path planner requires 'path' array in config")

        path = np.array(config["path"])
        if planner_type == PlannerType.BASE and path.shape[1] != 3:
            raise ValueError(
                f"Base path must be SE2 [x, y, yaw], got shape {path.shape}"
            )
        elif planner_type == PlannerType.EE and path.shape[1] != 6:
            raise ValueError(
                f"EE path must be SE3 [x, y, z, roll, pitch, yaw], got shape {path.shape}"
            )

        # Generate velocities (simple finite difference)
        velocities = np.zeros_like(path)
        if len(path) > 1:
            velocities[:-1] = np.diff(path, axis=0) / config.get("dt", 0.01)
            velocities[-1] = velocities[-2]  # Extend last velocity

        times = np.arange(len(path)) * config.get("dt", 0.01)

        self.plan = {
            "t": times,
            "p": path,
            "v": velocities,
        }

    def getTrackingPoint(self, t, robot_states=None):
        """Get tracking point from path."""
        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time
        p, v = interpolate(te, self.plan)
        return p, v

    def getTrackingPointArray(self, robot_states, num_pts, dt, time_offset=0):
        """Get array of tracking points along the path.

        Args:
            robot_states: Current robot states (unused, kept for interface compatibility)
            num_pts: Number of points to return
            dt: Time step between points
            time_offset: Time offset from current time

        Returns:
            (positions, velocities): Arrays of shape (num_pts, dim)
        """
        if len(self.plan["t"]) == 0 or len(self.plan["p"]) == 0:
            dim = 3 if self.type == PlannerType.BASE else 6
            return np.zeros((num_pts, dim)), np.zeros((num_pts, dim))

        # Get times for the array (relative to plan start)
        times = time_offset + np.arange(num_pts) * dt

        # Interpolate positions and velocities for all times at once
        positions = np.array([interpolate(t, self.plan)[0] for t in times])
        velocities = np.array([interpolate(t, self.plan)[1] for t in times])

        return positions, velocities

    def _compute_error(self, curr_pose, end_pose):
        """Compute position and orientation errors separately.

        Returns:
            tuple: (pos_err, ori_err) - position error in meters, orientation error in radians
        """
        if self.type == PlannerType.BASE:
            pos_err = np.linalg.norm(curr_pose[:2] - end_pose[:2])
            yaw_err = abs(wrap_pi_scalar(curr_pose[2] - end_pose[2]))
            return pos_err, yaw_err
        else:  # EE
            pos_err = np.linalg.norm(curr_pose[:3] - end_pose[:3])
            # Wrap Euler angles properly before computing orientation error
            ori_diff = wrap_pi_array(curr_pose[3:] - end_pose[3:])
            ori_err = np.linalg.norm(ori_diff)
            return pos_err, ori_err

    def checkFinished(self, t, curr_pose, curr_vel=None):
        """Check if path has been completed.

        Args:
            t: Current time
            curr_pose: Current pose array in world frame
                - For BASE: [x, y, yaw]
                - For EE: [x, y, z, roll, pitch, yaw]
            curr_vel: Current velocity array (optional, for end_stop check)
        """
        end_pose = self.plan["p"][-1]

        pos_err, ori_err = self._compute_error(curr_pose, end_pose)
        pos_cond = pos_err < self.tracking_err_tol
        ori_cond = ori_err < self.tracking_ori_err_tol
        pos_ori_cond = pos_cond and ori_cond
        vel_cond = curr_vel is not None and np.linalg.norm(curr_vel) < 1e-2

        if (not self.end_stop and pos_ori_cond) or (
            self.end_stop and pos_ori_cond and vel_cond
        ):
            if not self.finished:
                self.finished = True
                self.py_logger.info(
                    f"{self.name} finished (pos_err: {pos_err:.4f}, ori_err: {ori_err:.4f})"
                )

        return self.finished

    def reset(self):
        """Reset planner state."""
        self.finished = False
        self.started = False
        self.start_time = 0
        self.py_logger.info(f"{self.name} reset")


def create_planner(config: dict):
    """Create a planner instance from configuration.

    Args:
        config: Planner configuration dictionary containing "planner_type"

    Returns:
        Instance of the requested planner

    Raises:
        ValueError: If the planner type is unknown or missing
    """
    if "planner_type" not in config:
        raise ValueError("Configuration missing 'planner_type' field")

    planner_type = config["planner_type"]

    if planner_type == "WaypointPlanner":
        return WaypointPlanner(config)
    elif planner_type == "PathPlanner":
        return PathPlanner(config)
    else:
        raise ValueError(
            f"Unknown planner type: '{planner_type}'. "
            f"Available: WaypointPlanner, PathPlanner"
        )
