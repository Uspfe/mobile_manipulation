#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg
from liegroups import SE3, SO3

from mmseq_plan.PlanBaseClass import Planner
from mmseq_utils.transformation import *

class EESimplePlanner(Planner):
    def __init__(self, planner_params):
        self.name = planner_params["name"]
        self.target_pos = np.array(planner_params["target_pos"])
        self.type = "EE"
        self.ref_type = "waypoint"
        self.ref_data_type = "Vec3"

        self.finished = False
        self.reached_target = False
        self.t_reached_target = 0
        self.hold_period = planner_params["hold_period"]
        self.tracking_err_tol = planner_params["tracking_err_tol"]

        super().__init__()

    def getTrackingPoint(self, t, robot_states=None):
        return self.target_pos, np.zeros(3)
    
    def checkFinished(self, t, ee_curr_pos):
        if np.linalg.norm(ee_curr_pos - self.target_pos) > self.tracking_err_tol:
            if self.reached_target:
                self.reset()
            return self.finished

        if not self.reached_target:
            self.reached_target = True
            self.t_reached_target=t
            self.py_logger.info(self.name + " Planner Reached Target.")
            return self.finished

        if t - self.t_reached_target > self.hold_period:
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished.")

        return self.finished

    def reset(self):
        self.reached_target = False
        self.t_reached_target = 0
        self.finished = False
        self.py_logger.info(self.name + " Planner Reset.")


class EESimplePlannerRandom(Planner):
    def __init__(self, planner_params):
        self.target_pose_true = np.array(planner_params["target_pose"])
        self.regenerate_count = 0
        self.regenerate(0.75)
        self.type = "EE"
        self.ref_type = "pos"
        self.finished = False
        self.reached_target = False
        self.stamp = 0
        self.hold_period = planner_params["hold_period"]
        
    def getTrackingPoint(self, t, robot_states=None):
        if not self.finished:
            return self.target_pose, np.zeros(3)
        else:
            return None, None
    
    def checkFinished(self, t, current_EE):
        if self.regenerate_count == 9 and not self.reached_target and np.linalg.norm(current_EE - self.target_pose) < 0.015:
            self.reached_target = True
            self.stamp=t
            self.py_logger.info("Reached")
            
        elif self.reached_target and not self.finished:
            if t - self.stamp > self.hold_period:
                self.finished = True
                self.py_logger.info("Finished")
    def reset(self):
        self.reached_target = False
        self.stamp = 0
        self.finished = False
    
    def regenerate(self, sigma=1):
        while True:
            noise = np.random.randn(3) * sigma
            self.target_pose = self.target_pose_true +noise
            if self.target_pose[2] > 0:
                break
        self.regenerate_count +=1


class EEPosTrajectory(Planner):
    def __init__(self, planner_params):
        self.type = "EE"
        self.finished = False
        self.reached_target = False
        self.stamp = 0
        self.hold_period = planner_params["hold_period"]
        self.ref_type = "pose"

        super().__init__()
        
    def getTrackingPoint(self, t, robot_states=None):
        wp = np.array([np.sin(t), 0, np.sin(t)*np.cos(t)])
        disp = np.array([0., 2., 1.])
        
        return wp+disp, np.zeros(3)
        
    
    def checkFinished(self, t, state_ee):
        pass
    
    def reset(self):
        self.reached_target = False
        self.stamp = 0
        self.finished = False

class EESixDofWaypoint(Planner):
    def __init__(self, planner_params):
        self.target_pose = np.array(planner_params["target_pose"])
        self.type = "EE"
        self.finished = False
        self.reached_target = False
        self.stamp = 0
        self.hold_period = planner_params["hold_period"]
        self.ref_type = "pose"

        super().__init__()
        
    def getTrackingPoint(self, t, robot_states=None):
        if not self.finished:
            return self.target_pose, np.zeros(6)
        else:
            return self.target_pose, np.zeros(6)
    
    def checkFinished(self, t, state_ee):
        # state_ee a Homogeneous Transformation matrix
        Terr = np.matmul(linalg.inv(state_ee), self.target_pose)
        Terr = SE3(SO3(Terr[:3,:3]), Terr[:3, 3])
        twist = Terr.log()
        if not self.finished and np.linalg.norm(twist) > 0.2:
            self.reset()
        if not self.reached_target and np.linalg.norm(twist) < 0.1:
            self.reached_target = True
            self.stamp=t
            self.py_logger.info("Reached")
        elif self.reached_target and not self.finished:
            if t - self.stamp > self.hold_period:
                self.finished = True
                self.py_logger.info("Finished")
        
        # print("Target {}".format(self.target_pose))
        # print("Curret {}".format(state_ee))
    
    def reset(self):
        self.reached_target = False
        self.stamp = 0
        self.finished = False


if __name__ == '__main__':
    planner_params = {"target_pose": [0., 1.,], 'hold_period':1}
    
    planner = EESimplePlanner(planner_params)
    planner_params = {"target_pose": [0., 1., 1.], 'hold_period':1}
    planner = EESimplePlannerRandom(planner_params)
    
    T = make_trans_from_vec(np.array([0,0,1]) * np.pi/2, [1,0,0])
    
    planner_params = {"target_pose": T, 'hold_period': 0}
    planner = EESixDofWaypoint(planner_params)
    
    state_ee = make_trans_from_vec(np.array([0,0,1]) * np.pi/2*0.9, [1.,0,0])
    t = 0
    planner.checkFinished(t, state_ee)
    # sigma = 0.5
    # for i in range(6):
    #     sigma *= 0.5
    #     planner.regenerate(sigma)
    #     print(planner.target_pose)
