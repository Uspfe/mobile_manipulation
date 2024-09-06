#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging
from mmseq_utils.trajectory_generation import interpolate
class Planner(ABC):
    def __init__(self, name, type, ref_type, ref_data_type, frame_id):
        self.py_logger = logging.getLogger("Planner")
        self.name = name
        # The following variables are for automatically 
        # (1) publishing rviz visualization data
        # (2) assigning the correct mpc cost function
        self.type = type                        # base or EE
        self.ref_type = ref_type                # waypoint vs trajectory
        self.ref_data_type = ref_data_type      # Vec2 vs Vec3
        self.frame_id = frame_id                # base or EE


    @abstractmethod
    def getTrackingPoint(self, t, robot_states=None):
        """ get tracking point for controllers

        :param t: time (s)
        :type t: float
        :param robot_states: (joint angle, joint velocity), defaults to None
        :type robot_states: tuple, optional
        
        :return: position, velocity
        :rtype: numpy array, numpy array
        """
        p = None
        v = None
        return p,v
    
    @abstractmethod
    def checkFinished(self, t, P):
        """check if the planner is finished 

        :param t: time since the controller started
        :type t: float
        :param P: EE position for EE planner, base position for base planner
        :type P: numpy array
        :return: true if the planner has finished, false otherwise
        :rtype: boolean
        """
        finished = True
        return finished

class TrajectoryPlanner(Planner):

    def _interpolate(self, t, plan):
        p,v = interpolate(t, plan)

        return p, v