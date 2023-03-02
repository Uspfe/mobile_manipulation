#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import logging
import time
import sys

import numpy as np
import rospy
from pyb_utils.ghost import GhostSphere, GhostCylinder

from mmseq_control.HTMPC import HTMPC, HTMPCLex
from mmseq_simulator import simulation
from mmseq_plan.TaskManager import SoTStatic
from mmseq_utils import parsing
from mmseq_utils.logging import DataLogger, DataPlotter
from mobile_manipulation_central.ros_interface import MobileManipulatorROSInterface

def main():
    np.set_printoptions(precision=3, suppress=True)
    argv = rospy.myargv(argv=sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    parser.add_argument("--priority", type=str, default=None, help="priority, EE or base")
    parser.add_argument("--stmpctype", type=str, default=None,
                        help="STMPC type, SQP or lex. This overwrites the yaml settings")
    args = parser.parse_args(argv[1:])


    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    if args.stmpctype is not None:
        config["controller"]["type"] = args.stmpctype
    if args.priority is not None:
        config["planner"]["priority"] = args.priority

    if config["controller"]["type"] == "lex":
        config["controller"]["HT_MaxIntvl"] = 1

    ctrl_config = config["controller"]
    planner_config = config["planner"]

    if ctrl_config["type"] == "SQP" or ctrl_config["type"] == "SQP_TOL_SCHEDULE":
        controller = HTMPC(ctrl_config)
    elif ctrl_config["type"] == "lex":
        controller = HTMPCLex(ctrl_config)

    sot = SoTStatic(planner_config)

    # set py logger level
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    planner_log = logging.getLogger("Planner")
    planner_log.setLevel(config["logging"]["log_level"])
    planner_log.addHandler(ch)
    controller_log = logging.getLogger("Controller")
    controller_log.setLevel(config["logging"]["log_level"])
    controller_log.addHandler(ch)

    # TODO: How to organize logger for decentralized settings
    # init logger
    # logger = DataLogger(config)
    #
    # logger.add("sim_timestep", config["simulation"]["timestep"])
    # logger.add("duration", config["simulation"]["duration"])

    # logger.add("nq", sim_config["robot"]["dims"]["q"])
    # logger.add("nv", sim_config["robot"]["dims"]["v"])
    # logger.add("nx", sim_config["robot"]["dims"]["x"])
    # logger.add("nu", sim_config["robot"]["dims"]["u"])

    planners = sot.getPlanners(num_planners=2)

    # ROS related
    rospy.init_node("controller_ros")
    rate = rospy.Rate(ctrl_config["rate"])
    robot_interface = MobileManipulatorROSInterface()

    while not robot_interface.ready():
        robot_interface.brake()
        rate.sleep()

        if rospy.is_shutdown():
            return

    print("Controller received joint states. Proceed ... ")

    t = rospy.Time.now().to_sec()
    t0 = t
    acc = 0
    u = [0]
    while not rospy.is_shutdown():
        t1 = rospy.Time.now().to_sec()
        # print(t1)
        if t1 - t > (1./ ctrl_config["rate"])*5:
            print("Controller running slow. Last interval {}".format(t1 -t))
        t = t1

        # open-loop command
        robot_states = (robot_interface.q, robot_interface.v)
        # print("Msg Oldness Base: {}s, Arm: {}s".format(t - robot_interface.base.last_msg_time, t - robot_interface.arm.last_msg_time))
        # print("q: {}, v:{}, u: {}, acc:{}".format(robot_states[0][0], robot_states[1][0], u[0], acc))
        # tc1 = time.perf_counter()
        # tc1_ros = rospy.Time.now().to_sec()
        u, acc = controller.control(t-t0, robot_states, planners)
        # tc2_ros = rospy.Time.now().to_sec()
        # print("Controller Time (ROS): {}s ".format(tc2_ros - tc1_ros))
        # tc2 = time.perf_counter()
        # print(tc2 - tc1)

        robot_interface.publish_cmd_vel(u)
        rate.sleep()

    robot_interface.brake()

if __name__ == "__main__":
    main()
