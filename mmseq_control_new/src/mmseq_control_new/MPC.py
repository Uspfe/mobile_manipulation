from abc import ABC, abstractmethod
import time
import logging

import numpy as np
import casadi as cs
from spatialmath.base import rotz
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from mmseq_control.robot import MobileManipulator3D as MM
from mmseq_control.robot import CasadiModelInterface as ModelInterface
from mmseq_utils.math import wrap_pi_array
from mmseq_utils.casadi import casadi_sym_struct
from mmseq_control_new.MPCCostFunctions import EEPos3CostFunction, BasePos2CostFunction, ControlEffortCostFunction, EEPos3BaseFrameCostFunction, SoftConstraintsRBFCostFunction
from mmseq_control_new.MPCConstraints import SignedDistanceConstraint
import mobile_manipulation_central as mm
INF = 1e15

class MPC():
    def __init__(self, config):
        self.model_interface = ModelInterface(config)
        self.robot = self.model_interface.robot
        self.ssSymMdl = self.robot.ssSymMdl
        self.kinSymMdl = self.robot.kinSymMdls
        self.nx = self.ssSymMdl["nx"]
        self.nu = self.ssSymMdl["nu"]
        self.DoF = self.robot.DoF
        self.home = mm.load_home_position(config["home"])

        self.params = config
        self.dt = self.params["dt"]
        self.tf = self.params['prediction_horizon']
        self.N = int(self.tf / self.dt)
        self.QPsize = self.nx * (self.N + 1) + self.nu * self.N

        self.EEPos3Cost = EEPos3CostFunction(self.robot, config["cost_params"]["EEPos3"])
        self.EEPos3BaseFrameCost = EEPos3BaseFrameCostFunction(self.robot, config["cost_params"]["EEPos3"])
        self.BasePos2Cost = BasePos2CostFunction(self.robot, config["cost_params"]["BasePos2"])
        self.CtrlEffCost = ControlEffortCostFunction(self.robot, config["cost_params"]["Effort"])

        self.collision_link_names = ["self"] if self.params["self_collision_avoidance_enabled"] else []
        self.collision_link_names += self.model_interface.scene.collision_link_names["static_obstacles"] \
            if self.params["static_obstacles_collision_avoidance_enabled"] else []
        self.collision_link_names += ["sdf"] if self.params["sdf_collision_avoidance_enabled"] else []

        self.collisionCsts = {}
        for name in self.collision_link_names:
            sd_fcn = self.model_interface.getSignedDistanceSymMdls(name)
            sd_cst = SignedDistanceConstraint(self.robot, sd_fcn, 
                                              self.params["collision_safety_margin"][name], name)
            self.collisionCsts[name] = sd_cst
            
        self.collisionSoftCsts = {}
        for name,sd_cst in self.collisionCsts.items():
            expand = True if name !="sdf" else False
            self.collisionSoftCsts[name] = SoftConstraintsRBFCostFunction(self.params["collision_soft"][name]["mu"],
                                                                          self.params["collision_soft"][name]["zeta"],
                                                                          sd_cst, name+"CollisionSoftCst",
                                                                          expand=expand)
        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.x_bar[:, :self.DoF] = self.home
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.zopt = np.zeros(self.QPsize)  # current lineaization point
        self.u_prev = np.zeros(self.nu)
        self.xu_bar_init = False

        self.v_cmd = np.zeros(self.nx - self.DoF)

        self.py_logger = logging.getLogger("Controller")


    @abstractmethod
    def control(self, t, robot_states, planners, map=None):
        """

        :param t: current control time
        :param robot_states: (q, v) generalized coordinates and velocities
        :param planners: a list of planner instances
        :return: u, currently the best control inputs, aka, u_bar[0]
        """
        pass

    def _predictTrajectories(self, xo, u_bar):
        return MM.ssIntegrate(self.dt, xo, u_bar, self.ssSymMdl)

    def _getEEBaseTrajectories(self, x_bar):
        ee_bar = np.zeros((self.N + 1, 3))
        base_bar = np.zeros((self.N + 1, 3))
        for k in range(self.N+1):
            base_bar[k] = x_bar[k, :3]
            fee_fcn = self.kinSymMdl[self.robot.tool_link_name]
            ee_pos, ee_orn = fee_fcn(x_bar[k, :self.DoF])
            ee_bar[k] = ee_pos.toarray().flatten()

        return ee_bar, base_bar


    def reset(self):
        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.zopt = np.zeros(self.QPsize)  # current lineaization point
        self.xu_bar_init = False

        self.v_cmd = np.zeros(self.nx - self.DoF)

class STMPC(MPC):

    def __init__(self, config):
        super().__init__(config)
        self._construct()

    def _construct(self):
        # Construct AcadosModel
        model = AcadosModel()
        model.x = cs.MX.sym('x', self.nx)
        model.u = cs.MX.sym('u', self.nu)
        model.xdot = cs.MX.sym('xdot', self.nx)

        model.f_impl_expr = model.xdot - self.ssSymMdl["fmdl"](model.x, model.u)
        model.f_expl_expr = self.ssSymMdl["fmdl"](model.x, model.u)
        model.name = "MM"

        # get params from constraints
        num_terminal_cost = 2
        costs = [self.BasePos2Cost, self.EEPos3Cost, self.CtrlEffCost]
        costs += [cost for cost in self.collisionSoftCsts.values()]
        constraints = []
        self.p_dict = {}
        for cost in costs:
            self.p_dict.update(cost.get_p_dict())
        for cst in constraints:
            self.p_dict.update(cst.get_p_dict())
        self.p_struct = casadi_sym_struct(self.p_dict)
        print(self.p_struct)
        self.p_map = self.p_struct(0)
        model.p = self.p_struct.cat

        # Construct AcadosOCP
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.tf

        ocp.cost.cost_type = 'EXTERNAL'
        cost_expr = []
        for cost in costs:
            Ji = cost.J_fcn(model.x, model.u, cost.p_sym)
            cost_expr.append(Ji)
        ocp.model.cost_expr_ext_cost = sum(cost_expr)

        custom_hess_expr = []
        if self.params["use_custom_hess"]:
            for cost in costs:
                H_fcn = cost.get_custom_H_fcn()
                H_expr_i = H_fcn(model.x, model.u, cost.p_sym)
                custom_hess_expr.append(H_expr_i)
        ocp.model.cost_expr_ext_cost_custom_hess = sum(custom_hess_expr)

        # TODO: fix this. Terminal Cost function doesn't work for EE tracking
        # ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # cost_expr_e = sum(cost_expr[:num_terminal_cost])
        # cost_expr_e = cs.substitute(cost_expr_e, model.u, [])
        # fk_ee = self.robot.kinSymMdls[self.robot.tool_link_name]
        # Pee,_ = fk_ee(model.x[:9])
        # ocp.model.cost_y_expr_e = Pee
        # ocp.cost.W_e = np.eye(3) * self.params["cost_params"]["EEPos3"]["P"]
        # ocp.cost.yref_e = np.zeros(3)

        # control input constraints
        ocp.constraints.lbu = np.array(self.ssSymMdl["lb_u"])
        ocp.constraints.ubu = np.array(self.ssSymMdl["ub_u"])
        ocp.constraints.idxbu = np.arange(self.nu)

        # state constraints
        ocp.constraints.lbx = np.array(self.ssSymMdl["lb_x"])
        ocp.constraints.ubx = np.array(self.ssSymMdl["ub_x"])
        ocp.constraints.idxbx = np.arange(self.nx)

        # ocp.constraints.lbx_e = np.array(self.ssSymMdl["lb_x"])
        # ocp.constraints.ubx_e = np.array(self.ssSymMdl["ub_x"])
        # ocp.constraints.idxbx_e = np.arange(self.nx)

        # nonlinear constraints
        # TODO: what about the initial and terminal shooting nodes.
        h_expr_list = []
        for cst in constraints:
            h_expr_list.append(cst.g_fcn(model.x, model.u, cst.p_sym))
        
        if len(h_expr_list) > 0:
            h_expr = cs.vertcat(*h_expr_list)
            print(h_expr)
            model.con_h_expr = h_expr
            h_expr_num = h_expr.shape[0]
            ocp.constraints.uh = np.zeros(h_expr_num)
            ocp.constraints.lh = -INF*np.ones(h_expr_num)

        # TODO: slack variables?

        # initial condition
        ocp.constraints.x0 = self.x_bar[0]

        ocp.parameter_values = self.p_map.cat.full().flatten()

        # set options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = 'IRK'
        # ocp.solver_options.print_level = 1
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
        ocp.solver_options.qp_solver_iter_max = 100
        ocp.solver_options.nlp_solver_tol_stat = 1e-3
        # Construct AcadosOCPSolver
        ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_stmpc.json')

        self.model = model
        self.ocp = ocp
        self.ocp_solver = ocp_solver

    def control(self, t, robot_states, planners, map=None):

        self.py_logger.debug("control time {}".format(t))
        self.curr_control_time = t
        q, v = robot_states
        q[2:9] = wrap_pi_array(q[2:9])
        xo = np.hstack((q, v))

        # 0.1 Get warm start point
        self.u_bar[:-1] = self.u_bar[1:]
        self.u_bar[-1] = 0
        self.x_bar = self._predictTrajectories(xo, self.u_bar)            


        # 0.2 Get ref, sdf map,
        r_bar_map = {}
        self.ree_bar = []
        self.rbase_bar = []
        for planner in planners:
            r_bar = [planner.getTrackingPoint(t + k * self.dt, (self.x_bar[k, :self.DoF], self.x_bar[k, self.DoF:]))[0]
                        for k in range(self.N + 1)]
            acceptable_ref = True
            if planner.type == "EE":
                if planner.ref_data_type == "Vec3":
                    r_bar_map["EEPos3"] = r_bar
                else:
                    acceptable_ref = False
            elif planner.type == "base":
                if planner.ref_data_type == "Vec2":
                    r_bar_map["BasePos2"] = r_bar
                else:
                    acceptable_ref = False

            if not acceptable_ref:
                self.py_logger.warning(f"unknown cost type {planner.ref_data_type}, planner {planner.name}")
            
            if planner.type == "EE":
                self.ree_bar = r_bar 
            elif planner.type == "base":
                self.rbase_bar = r_bar

        curr_p_map = self.p_struct(0)
    
        if map is not None:
            self.model_interface.sdf_map.update_map(*map)

        if self.params["sdf_collision_avoidance_enabled"]:
            params = self.model_interface.sdf_map.get_params()
            curr_p_map["x_grid_sdf"] = params[0]
            curr_p_map["y_grid_sdf"] = params[1]
            if self.model_interface.sdf_map.dim == 3:
                curr_p_map["z_grid_sdf"] = params[2]
                curr_p_map["value_sdf"] = params[3]
            else:
                curr_p_map["value_sdf"] = params[2]
        

        for i in range(self.N+1):
            # set initial guess
            self.ocp_solver.set(i, 'x', self.x_bar[i])

            # set parameters for tracking cost functions
            p_keys = self.p_struct.keys()
            for (name, r_bar) in r_bar_map.items():
                p_name_r = "_".join(["r", name])
                p_name_W = "_".join(["W", name])

                if p_name_r in p_keys:
                    # set reference
                    curr_p_map[p_name_r] = r_bar[i]

                    # Set weight matricies, assuming identity matrix with identical diagonal terms
                    if i == self.N:
                        curr_p_map[p_name_W] = self.params["cost_params"][name]["P"] * np.eye(r_bar[i].size)
                    else:
                        curr_p_map[p_name_W] = self.params["cost_params"][name]["Qk"] * np.eye(r_bar[i].size)
                else:
                    self.py_logger.warning(f"unknown p name {p_name_r}")

            self.ocp_solver.set(i, 'p', curr_p_map.cat.full().flatten())

        self.ocp_solver.solve_for_x0(xo, fail_on_nonzero_status=False)
        self.ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")
        self.solver_status = self.ocp_solver.status

        if self.solver_status !=0:
            for i in range(self.N):
                print(f"stage {i}: x: {self.ocp_solver.get(i, 'x')}")
                print(f"stage {i}: u: {self.ocp_solver.get(i, 'u')}")
                        
            for i in range(self.N):
                print(f"stage {i}: lam: {self.ocp_solver.get(i, 'lam')}")
            
            for i in range(self.N):
                print(f"stage {i}: pi: {self.ocp_solver.get(i, 'pi')}")

            if self.params["raise_exception_on_failure"]:
                raise Exception(f'acados acados_ocp_solver returned status {self.solver_status}')


        # get solution
        self.u_prev = self.u_bar[0].copy()
        for i in range(self.N):
            self.x_bar[i,:] = self.ocp_solver.get(i, "x")
            self.u_bar[i,:] = self.ocp_solver.get(i, "u")
        self.x_bar[self.N,:] = self.ocp_solver.get(self.N, "x")

        self.v_cmd = self.x_bar[0][self.robot.DoF:].copy()

        self.ee_bar, self.base_bar = self._getEEBaseTrajectories(self.x_bar)
        print(f"v{self.v_cmd}")
        print(f"u: {self.u_bar[0]}")
        return self.v_cmd, self.u_prev, self.u_bar.copy()



if __name__ == "__main__":
    # robot mdl
    from mmseq_utils import parsing
    config = parsing.load_config(
        "/home/tracy/Projects/mm_slam/mm_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml")

    STMPC(config["controller"])
