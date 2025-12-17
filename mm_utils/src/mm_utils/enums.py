"""Enums for planner and control system types."""

from enum import Enum


class PlannerType(Enum):
    """Type of planner: base or end-effector."""

    BASE = "base"
    EE = "EE"


class RefType(Enum):
    """Type of reference: waypoint or path."""

    WAYPOINT = "waypoint"
    PATH = "path"


class RefDataType(Enum):
    """Data type for references: SE2 (x,y,yaw) or SE3 (x,y,z,roll,pitch,yaw)."""

    SE2 = "SE2"
    SE3 = "SE3"


class FrameID(Enum):
    """Frame identifier: base or end-effector."""

    BASE = "base"
    EE = "EE"
    BASE_LINK = "base_link"  # For base frame in world coordinates
