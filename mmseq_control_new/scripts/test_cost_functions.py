import numpy as np
from mmseq_control.robot import MobileManipulator3D
from mmseq_control_new.MPCCostFunctions import BasePos2CostFunction,EEPos3CostFunction, ControlEffortCostFunction
from mmseq_utils.casadi_struct import casadi_sym_struct
import casadi as cs

if __name__ == "__main__":
    dt = 0.1
    N = 10
    # robot mdl
    from mmseq_utils import parsing

    config = parsing.load_config(
        "/home/tracy/Projects/mm_slam/mm_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml")
    robot = MobileManipulator3D(config["controller"])

    cost_base = BasePos2CostFunction(robot, config["controller"]["cost_params"]["BasePos2"])
    cost_ee = EEPos3CostFunction(robot, config["controller"]["cost_params"]["EEPos3"])
    cost_eff = ControlEffortCostFunction(robot, config["controller"]["cost_params"]["Effort"])
    cost_fcn = cost_eff

    q = [0.0, 0.0, 0.0] + [0.0, -2.3562, -1.5708, -2.3562, -1.5708, 1.5708]
    v = np.zeros(9)
    x = np.hstack((np.array(q), v))
    u = np.zeros(9)

    p_map_base = cost_base.p_struct(0)
    p_map_base['W'] = config["controller"]["cost_params"]["BasePos2"]["Qk"]
    p_map_base['r'] = np.array([1,0])
    J_base = cost_base.evaluate(x, u, p_map_base.cat)
    print(J_base)
    print(p_map_base)
    
    p_map_ee = cost_ee.p_struct(0)
    p_map_ee['W'] = config["controller"]["cost_params"]["EEPos3"]["Qk"]
    p_map_ee['r'] = np.array([1.194, 0.374, 1.596])
    J_ee = cost_ee.evaluate(x, u, p_map_ee.cat)
    print(J_ee)

    J_eff = cost_eff.evaluate(x, u, [])
    print(J_eff)

    H_eqn, _ = cs.hessian(cost_fcn.J_eqn, cs.veccat(cost_fcn.u_sym, cost_fcn.x_sym))
    H_fcn = cs.Function('H', [cost_fcn.x_sym, cost_fcn.u_sym, cost_fcn.p_sym], [H_eqn])

    H_val = H_fcn(x, u, cost_fcn.p_struct(0)).toarray()
    H_approx_val = cost_fcn.H_approx_fcn(x, u, cost_fcn.p_struct(0)).toarray()
    print(f"Hess Eval{H_val}")
    print(f"Hess Approx Eval{H_approx_val}")
    print(f"Diff{np.linalg.norm(H_val - H_approx_val)}")