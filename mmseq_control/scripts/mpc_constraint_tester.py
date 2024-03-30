import rospy
import numpy as np
import casadi as cs
import os
from mmseq_control.MPCConstraints import SignedDistanceCollisionConstraint
from mmseq_control.map import SDF2D
from mobile_manipulation_central.ros_interface import MapInterface
from cbf_mpc.barrier_function2 import CBF, CBFJacobian


def testSDFMapConstraint(config):
    tsdf_map_interface = MapInterface(topic_name="/pocd_slam_node/occupied_ef_nodes")
    map = SDF2D()
    sd_fcn = CBF('sdf', map)
    rate = rospy.Rate(10)
    while not tsdf_map_interface.ready() and not rospy.is_shutdown():
        rate.sleep()

    status, tsdf = tsdf_map_interface.get_map()
    map.update_map(tsdf)

    from mmseq_control.robot import  CasadiModelInterface
    from mmseq_control.MPCCostFunctions import SoftConstraintsRBFCostFunction
    casadi_model_interface = CasadiModelInterface(config["controller"])
    casadi_model_interface.sdf_map.update_map(tsdf)

    dt = 0.1
    N = 10
    robot_mdl = casadi_model_interface.robot

    sd_eqn_base = sd_fcn(robot_mdl.qb_sym[:2])
    sd_fcn_base = cs.Function("sdf_base", [robot_mdl.q_sym], [sd_eqn_base])
    # const = SignedDistanceCollisionConstraint(robot_mdl, sd_fcn_base, dt, N, 0.6, "sdf_base")
    const = SignedDistanceCollisionConstraint(robot_mdl, casadi_model_interface.getSignedDistanceSymMdls("sdf"), dt, N, 0.0, "sdf_base")

    nx = robot_mdl.ssSymMdl["nx"]
    nu = robot_mdl.ssSymMdl["nu"]
    x_bar = np.ones((N + 1, nx)) * 0.0
    u_bar = np.ones((N, nu)) * 0
    x_bar[:, 0] = np.arange(N + 1)*0.2
    print(const.g_fcn(x_bar.T, u_bar.T))

    mu = config["controller"]["collision_soft"]["mu"]
    zeta = config["controller"]["collision_soft"]["zeta"]
    const_soft = SoftConstraintsRBFCostFunction(mu, zeta, const, "SelfCollisionSoftConstraint")
    J_soft = const_soft.evaluate(x_bar, u_bar)
    print(J_soft)

if __name__ == "__main__":
    rospy.init_node("map_tester")

    dt = 0.1
    N = 10
    # robot mdl
    from mmseq_utils import parsing
    config = parsing.load_config("/home/tracy/Projects/mm_slam/mm_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml")
    testSDFMapConstraint(config)