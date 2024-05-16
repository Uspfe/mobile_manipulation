import numpy as np
import rosbag
import rospy
import matplotlib.pyplot as plt

from mmseq_control.robot import MobileManipulator3D
from mmseq_control_new.MPCConstraints import HierarchicalTrackingConstraint, SignedDistanceConstraint
from mmseq_control_new.MPCCostFunctions import BasePos2CostFunction, EEPos3CostFunction, SoftConstraintsRBFCostFunction

def test_HierarchicalTrackingConstraint(config):
    robot = MobileManipulator3D(config["controller"])

    q = [0.0, 0.0, 0.0] + [0.0, -2.3562, -1.5708, -2.3562, -1.5708, 1.5708]
    v = np.zeros(9)
    x = np.hstack((np.array(q), v))
    u = np.zeros(9)

    cost_base = BasePos2CostFunction(robot, config["controller"]["cost_params"]["BasePos2"])
    cost_ee = EEPos3CostFunction(robot, config["controller"]["cost_params"]["EEPos3"])
    test_cst = HierarchicalTrackingConstraint(cost_base, "base_hierarchy")

    param_val = test_cst.p_struct(0)
    # param_val['r_EEPos3'] = np.ones(3)
    # param_val['e_p'] = np.ones(3)*0.5
    param_val['r_BasePos2'] = np.ones(2)
    param_val['e_p'] = np.ones(2)*0.5
                            
    print(test_cst.check(x, u, param_val.cat))
    print(test_cst.get_p_dict())


def test_SignedDistanceConstraint(config):

    from mobile_manipulation_central.ros_interface import MapInterfaceNew
    map_ros_interface = MapInterfaceNew(config["controller"])
    rate = rospy.Rate(100)

    while not map_ros_interface.ready() and not rospy.is_shutdown():
        rate.sleep()

    from mmseq_control.robot import  CasadiModelInterface
    casadi_model_interface = CasadiModelInterface(config["controller"])
    while not rospy.is_shutdown():
        _, map = map_ros_interface.get_map()
        casadi_model_interface.sdf_map.update_map(*map)
        tsdf, tsdf_vals = map_ros_interface.tsdf, map_ros_interface.tsdf_vals
        # query points inside the tsdf points
        pts = np.around(np.array([np.array([p.x,p.y]) for p in tsdf]), 2).reshape((len(tsdf),2))
        xs = pts[:,0]
        ys = pts[:,1]
        x_lim = [min(xs), max(xs)]
        y_lim = [min(ys), max(ys)]
        # Plot sdf map
        casadi_model_interface.sdf_map.vis(x_lim=x_lim,
                                        y_lim=y_lim,
                                        block=False)

        qbs_x = np.linspace(x_lim[0], x_lim[1], int(1.0/0.1 * (x_lim[1] - x_lim[0]))+1)
        qbs_y = np.linspace(y_lim[0], y_lim[1], int(1.0/0.1 * (y_lim[1] - y_lim[0]))+1)
        X,Y= np.meshgrid(qbs_x, qbs_y)

        robot_mdl = casadi_model_interface.robot
        N = X.size
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        x = np.ones((nx,N)) * 0
        x[0, :] = X.flatten()
        x[1, :] = Y.flatten()
        u = np.ones((nu,N)) * 0
        d_safe = 0.15
        const = SignedDistanceConstraint(robot_mdl, casadi_model_interface.getSignedDistanceSymMdls("sdf"), d_safe, "sdf_2d")
        params = casadi_model_interface.sdf_map.get_params()
        param_map = const.p_struct(0)
        param_map["x_grid"] = params[0]
        param_map["y_grid"] = params[1]
        param_map["value"] = params[2]
        g_sdf = const.check(x, u, param_map.cat).toarray()
        g_sdf = g_sdf.flatten().reshape(X.shape)
        print(const.get_p_dict())

        # Plot collision constraint
        fig_g, ax_g = plt.subplots()
        levels = np.linspace(-2., 0.5, int(2.5/0.25)+1)
        cs = ax_g.contour(X,Y,g_sdf, levels)
        ax_g.clabel(cs, levels)
        ax_g.grid()
        ax_g.set_title("Collision Constraint $g = -(sd(x) - d_{safe})$, " + "$d_{safe} = $" + f"{0.6 + d_safe}m")   # 0.6 is base collision radius
        ax_g.set_xlabel("x(m)")
        ax_g.set_ylabel("y(m)")
        plt.show(block=False)


        mu = config["controller"]["collision_soft"]['sdf']["mu"]
        zeta = config["controller"]["collision_soft"]['sdf']["zeta"]
        print(const.get_p_dict())
        const_soft = SoftConstraintsRBFCostFunction(mu, zeta, const, "SelfCollisionSoftConstraint",expand=False)
        # J_soft = [const_soft.evaluate(x_bar[i,:], u_bar)/X.size for i in range(N+1)]
        J_soft = const_soft.evaluate_vec(x, u, param_map.cat)
        J_soft = np.array(J_soft).reshape(X.shape)

        fig_J, ax_J = plt.subplots()
        levels = np.linspace(-0.5, 5, int(5.5/0.5)+1)
        cs = ax_J.contour(X,Y,J_soft, levels)
        ax_J.clabel(cs, levels)
        ax_J.grid()
        ax_J.set_title("Soft Collision Constraint Cost $\mathcal{J} = RBF(g)(x))$, " + "$d_{safe} = $" + f"{0.6 + d_safe}m")   # 0.6 is base collision radius
        ax_J.set_xlabel("x(m)")
        ax_J.set_ylabel("y(m)")
        plt.show()


if __name__ == "__main__":
    # robot mdl
    from mmseq_utils import parsing
    config = parsing.load_config(
        "/home/tracy/Projects/mm_slam/mm_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml")
    rospy.init_node("constraints_tester")

    test_SignedDistanceConstraint(config)
    # test_HierarchicalTrackingConstraint(config)
    