"""TaskManager - simplified sequential task execution."""

import logging

from mm_plan.Planners import create_planner
from mm_utils.enums import PlannerType


class TaskManager:
    """Simplified task manager that executes planners sequentially."""

    def __init__(self, config):
        self.config = config
        self.started = False
        self.planners = [create_planner(task) for task in config["tasks"]]
        self.planner_num = len(self.planners)
        self.curr_task_id = 0
        self.logger = logging.getLogger("Planner")

    def activatePlanners(self):
        """Activate all planners."""
        for planner in self.planners:
            planner.activate()

    def getPlanners(self, num_planners=2):
        """Get the active planners (current task and next if available).

        Args:
            num_planners: Number of planners to return (default 2)

        Returns:
            List of active planners
        """
        end_id = min(self.curr_task_id + num_planners, self.planner_num)
        start_id = max(0, end_id - num_planners)
        return self.planners[start_id:end_id]

    def update(self, t, states):
        """Update task manager and check if current task is finished.

        Args:
            t: Current time
            states: Dictionary with "EE" and "base" states
                - "base": {"pose": [x, y, yaw], "velocity": [vx, vy, vyaw]}
                - "EE": {"pose": [x, y, z, roll, pitch, yaw], "velocity": [vx, vy, vz, wx, wy, wz]}

        Returns:
            (finished, increment): Whether current task finished, and increment (always 1)
        """
        if self.curr_task_id >= self.planner_num:
            return True, 1

        planner = self.planners[self.curr_task_id]
        finished = False

        if planner.type == PlannerType.EE:
            pose = states["EE"]["pose"]
            vel = states["EE"].get("velocity")
            finished = planner.checkFinished(t, pose, vel)
        elif planner.type == PlannerType.BASE:
            pose = states["base"]["pose"]
            vel = states["base"].get("velocity")
            finished = planner.checkFinished(t, pose, vel)

        if finished:
            if self.curr_task_id < self.planner_num - 1:
                self.curr_task_id += 1
                self.logger.info(
                    "Task finished, moving to task %d/%d",
                    self.curr_task_id + 1,
                    self.planner_num,
                )
            else:
                self.logger.info(
                    "All tasks completed (%d/%d)", self.planner_num, self.planner_num
                )
            return True, 1

        return False, 1

    def print(self):
        """Print task manager status."""
        print(
            f"TaskManager: {self.planner_num} tasks, currently on task {self.curr_task_id + 1}"
        )
