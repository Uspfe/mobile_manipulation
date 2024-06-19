import sys

from omni.isaac.kit import SimulationApp
# URDF import, configuration and simulation sample
simulation_app = SimulationApp({"headless": False})

import numpy as np
from mmseq_utils import parsing
import rospy
import yaml
import argparse

from omni.isaac.core.utils import extensions
extensions.enable_extension("omni.isaac.ros_bridge")
from mmseq_sim_isaac.isaac_sim_env import IsaacSimEnv
from omni.isaac.core.utils.rotations import euler_to_rot_matrix

def main():
    parser = argparse.ArgumentParser()
    argv = rospy.myargv(argv=sys.argv)

    parser.add_argument("--config", required=True, help="Path to configuration file.")
    args = parser.parse_args(argv[1:])
    
    config = parsing.load_config(args.config)
    sim_config = config["simulation"]

    rospy.init_node("isaac_sim_ros")

    sim = IsaacSimEnv(sim_config)
    robot = sim.robot
    world = sim.world
    robot_ros_interface = sim.robot_ros_interface

    # disable gravity to use joint velocity control
    robot.disable_gravity()

    # if no cmd_vel comes, brake
    while not robot_ros_interface.ready() and simulation_app.is_running():
        sim.publish_feedback()
        sim.apply_joint_velocities()

        sim.step(render=True)
        sim.publish_ros_topics()

    while simulation_app.is_running():
        t = world.current_time

        q = robot.get_joint_positions()
        v = robot.get_joint_velocities()

        # convert base cmd_vel from base frame to world frame
        base_cmd_vel_b = robot_ros_interface.base.cmd_vel
        R_wb = euler_to_rot_matrix([0,0,q[2]], extrinsic=False)
        base_cmd_vel = R_wb @ base_cmd_vel_b
        sim.apply_joint_velocities(np.concatenate((base_cmd_vel, robot_ros_interface.arm.cmd_vel)))

        robot_ros_interface.publish_feedback(t=t, q=q, v=v)
        
        sim.step(render=True)
        sim.publish_ros_topics()

    simulation_app.close()

if __name__ == "__main__":
    main()