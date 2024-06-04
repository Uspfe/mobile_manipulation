import numpy as np
import casadi as cs
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from mmseq_utils.math import wrap_pi_array
from mmseq_utils.casadi_struct import casadi_sym_struct
from mmseq_control_new.MPCConstraints import HierarchicalTrackingConstraint
from mmseq_control_new.MPCCostFunctions import CostFunctions
import mobile_manipulation_central as mm

from mmseq_control_new.MPC import MPC, INF

class HTMPC(MPC):

    def __init__(self, config):
        super().__init__(config)
        self.EEPos3LexConstraint = HierarchicalTrackingConstraint(self.EEPos3Cost, "_".join([self.EEPos3Cost.name, "Lex"]))
        self.BasePos2LexConstraint = HierarchicalTrackingConstraint(self.BasePos2Cost, "_".join([self.BasePos2Cost.name, "Lex"]))
        
        ocp1, ocp_solver1, p_struct1 = self._construct([self.EEPos3Cost, self.RegularizationCost], [], "EEPos3")
        ocp2, ocp_solver2, p_struct2 = self._construct([self.BasePos2Cost, self.CtrlEffCost], [self.EEPos3LexConstraint], "BasePos2")
        
        self.stmpcs = [ocp1, ocp2]
        self.stmpc_solvers = [ocp_solver1, ocp_solver2]
        self.stmpc_p_structs = [p_struct1, p_struct2]

    def _construct(self, costs, constraints, name="MM"):
        # Construct AcadosModel
        model = AcadosModel()
        model.x = cs.MX.sym('x', self.nx)
        model.u = cs.MX.sym('u', self.nu)
        model.xdot = cs.MX.sym('xdot', self.nx)

        model.f_impl_expr = model.xdot - self.ssSymMdl["fmdl"](model.x, model.u)
        model.f_expl_expr = self.ssSymMdl["fmdl"](model.x, model.u)
        model.name = name

        # get params from constraints
        num_terminal_cost = 2
        costs += [cost for cost in self.collisionSoftCsts.values()]
        p_dict = {}
        for cost in costs:
            p_dict.update(cost.get_p_dict())
        for cst in constraints:
            p_dict.update(cst.get_p_dict())
        p_struct = casadi_sym_struct(p_dict)
        print(p_struct)
        p_map = p_struct(0)
        model.p = p_struct.cat

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
        # ocp.cost.cost_type_e = 'EXTERNAL'
        # cost_expr_e = sum(cost_expr[:num_terminal_cost])
        # cost_expr_e = cs.substitute(cost_expr_e, model.u, [])
        # ocp.model.cost_expr_ext_cost_e = cost_expr_e
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
            ocp.model.con_h_expr = h_expr
            h_expr_num = h_expr.shape[0]
            ocp.constraints.uh = np.zeros(h_expr_num)
            ocp.constraints.lh = -INF*np.ones(h_expr_num)

            ocp.model.con_h_expr_e = cs.substitute(h_expr, model.u, [])
            ocp.constraints.uh_e = np.zeros(h_expr_num)
            ocp.constraints.lh_e = -INF*np.ones(h_expr_num)

            # ocp.constraints.idxsh = np.arange(h_expr_num)
            # ocp.constraints.idxsh_e = np.arange(h_expr_num)

            # ocp.cost.Zu = np.eye(h_expr_num)*1e3
            # ocp.cost.Zl = np.eye(h_expr_num)*0
            # ocp.cost.zl = np.ones(h_expr_num)*0
            # ocp.cost.zu = np.ones(h_expr_num)
        
            # ocp.cost.Zu_e = np.eye(h_expr_num)*1e3
            # ocp.cost.Zl_e = np.eye(h_expr_num)*0
            # ocp.cost.zl_e = np.ones(h_expr_num)*0
            # ocp.cost.zu_e = np.ones(h_expr_num)


        # self.h_fcn = cs.Function('h', [model.x, model.u, model.p], [h_expr])
        # TODO: slack variables?

        # initial condition
        ocp.constraints.x0 = self.x_bar[0]
        # default parameters
        # TODO: is there a better way to assign initial parameters. Now they're all zeros
        ocp.parameter_values = p_map.cat.full().flatten()

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
        ocp.solver_options.qp_solver_warm_start = True
        # Construct AcadosOCPSolver
        ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_stmpc.json')

        return ocp, ocp_solver, p_struct

    def control(self, t, robot_states, planners, map=None):
        task_num = len(planners)
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

        if map is not None:
            self.model_interface.sdf_map.update_map(*map)

        for pid, planner in enumerate(planners):
            r_bar = [planner.getTrackingPoint(t + k * self.dt, (self.x_bar[k, :self.DoF], self.x_bar[k, self.DoF:]))[0]
                        for k in range(self.N + 1)]
            acceptable_ref = False
            if planner.type == "EE" and  planner.ref_data_type == "Vec3":
                acceptable_ref = True
                r_bar_map[self.EEPos3Cost.name] = r_bar
            elif planner.type == "base" and planner.ref_data_type == "Vec2":
                acceptable_ref = True
                r_bar_map[self.BasePos2Cost.name] = r_bar

            if not acceptable_ref:
                self.py_logger.warning(f"unknown cost type {planner.ref_data_type}, planner {planner.name}")
            
            if planner.type == "EE":
                self.ree_bar = r_bar 
            elif planner.type == "base":
                self.rbase_bar = r_bar

        # Optimal tracking error
        e_p_bar_map = {}

        # for logging
        self.cost_iter = np.zeros((task_num, 2))
        self.cost_final = np.zeros(task_num)
        self.solver_status = np.zeros(task_num)
        self.step_size = np.zeros(task_num)
        self.nlp_iter = np.zeros(task_num)

        tracking_cost_fcns = []
        for task_id, (stmpc, stmpc_solver, p_struct) in enumerate(zip(self.stmpcs, self.stmpc_solvers, self.stmpc_p_structs)):
            tracking_cost_fcn_name = stmpc.name
            tracking_cost_fcn = self.EEPos3Cost if tracking_cost_fcn_name == "EEPos3" else self.BasePos2Cost
            tracking_cost_fcns.append(tracking_cost_fcn)

            if self.params["sdf_collision_avoidance_enabled"]:
                sdf_params = self.model_interface.sdf_map.get_params()

            curr_p_map_bar = []
            for k in range(self.N+1):
                # set parameters
                curr_p_map = p_struct(0)

                # Set regularization
                if task_id == 0:
                    curr_p_map["eps_Regularization"] = self.params["cost_params"]["Regularization"]["eps"]

                # SDF parameters
                if self.params["sdf_collision_avoidance_enabled"]:
                    curr_p_map["x_grid_sdf"] = sdf_params[0]
                    curr_p_map["y_grid_sdf"] = sdf_params[1]
                    if self.model_interface.sdf_map.dim == 3:
                        curr_p_map["z_grid_sdf"] = sdf_params[2]
                        curr_p_map["value_sdf"] = sdf_params[3]
                    else:
                        curr_p_map["value_sdf"] = sdf_params[2]

                # set initial guess
                stmpc_solver.set(k, 'x', self.x_bar[k])

                # set parameters for tracking cost functions
                p_keys = p_struct.keys()
                p_name_r = "_".join(["r", tracking_cost_fcn_name])

                if p_name_r in p_keys:
                    r_k = r_bar_map[tracking_cost_fcn_name][k]
                    # set reference
                    curr_p_map[p_name_r] = r_k
                    p_name_W = "_".join(["W", tracking_cost_fcn_name])

                    # Set weight matricies, assuming identity matrix with identical diagonal terms
                    if k == self.N:
                        curr_p_map[p_name_W] = self.params["cost_params"][tracking_cost_fcn_name]["P"] * np.eye(r_k.size)
                    else:
                        curr_p_map[p_name_W] = self.params["cost_params"][tracking_cost_fcn_name]["Qk"] * np.eye(r_k.size)

                else:
                    self.py_logger.warning(f"unknown p name {p_name_r} in {tracking_cost_fcn_name}'s p_struct")
                
                for p in range(task_id):
                    # set parameters for lexicographic optimality constraints
                    name = self.stmpcs[p].name
                    p_name_e_p = "_".join(["e_p", name, "Lex"])
                    curr_p_map[p_name_e_p] = e_p_bar_map[name][k]

                    p_name_r = "_".join(["r", name])
                    r_k = r_bar_map[name][k]
                    # set reference
                    curr_p_map[p_name_r] = r_k

                stmpc_solver.set(k, 'p', curr_p_map.cat.full().flatten())
                curr_p_map_bar.append(curr_p_map)

            # Log: initial cost
            self.cost_iter[task_id, 0] = self.evaluate_cost_function(tracking_cost_fcn, self.x_bar, self.u_bar, curr_p_map_bar)

            # Solve stmpc
            stmpc_solver.solve_for_x0(xo, fail_on_nonzero_status=False)
            stmpc_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

            if stmpc_solver.status !=0:
                for i in range(self.N):
                    print(f"stage {i}: x: {stmpc_solver.get(i, 'x')}")
                    print(f"stage {i}: u: {stmpc_solver.get(i, 'u')}")
                            
                for i in range(self.N):
                    print(f"stage {i}: lam: {stmpc_solver.get(i, 'lam')}")
                
                for i in range(self.N):
                    print(f"stage {i}: pi: {stmpc_solver.get(i, 'pi')}")

                if self.params["raise_exception_on_failure"]:
                    raise Exception(f'acados acados_ocp_solver returned status {stmpc_solver.status}')
                
            # get solution
            for i in range(self.N):
                self.x_bar[i,:] = stmpc_solver.get(i, "x")
                self.u_bar[i,:] = stmpc_solver.get(i, "u")
            self.x_bar[self.N,:] = stmpc_solver.get(self.N, "x")

            # for i in range(self.N):
            #     print(f"time {i}: h: {self.h_fcn(self.x_bar[i], self.u_bar[i], curr_p_map.cat.full().flatten())}")

            # get e_p_bar
            e_p_bar_map[tracking_cost_fcn_name] = []
            for k in range(self.N+1):
                # Relaxed Lex Constraints |e_k| \leq |e^*_k| + eps
                e_p_bar_map[tracking_cost_fcn_name].append(np.abs(tracking_cost_fcn.get_e(self.x_bar[k],
                                                            [],
                                                            r_bar_map[tracking_cost_fcn_name][k])) + 0.0025)
                print(f"task:{tracking_cost_fcn_name}, time{k}: e_p {e_p_bar_map[tracking_cost_fcn_name][-1]}")
            
            e_p_bar_map[tracking_cost_fcn_name] = np.array(e_p_bar_map[tracking_cost_fcn_name])
            print(f"cost {tracking_cost_fcn_name}: {stmpc_solver.get_cost()}")

            # Log: solver status, step size, cost after stmpc
            self.solver_status[task_id] = stmpc_solver.status
            self.step_size[task_id] = np.mean(stmpc_solver.get_stats('alpha'))
            self.cost_iter[task_id, 1] = self.evaluate_cost_function(tracking_cost_fcn, self.x_bar, self.u_bar, curr_p_map_bar)
            self.nlp_iter[task_id] = stmpc_solver.get_stats('sqp_iter')

        # Log: final cost for each task after all stmpcs have been solved
        for i, tracking_cost_fcn in enumerate(tracking_cost_fcns):
            self.cost_final[i] = self.evaluate_cost_function(tracking_cost_fcn, 
                                                             self.x_bar,
                                                             self.u_bar,
                                                             curr_p_map_bar)

        self.v_cmd = self.x_bar[0][self.robot.DoF:].copy()
        self.ee_bar, self.base_bar = self._getEEBaseTrajectories(self.x_bar)

        return self.v_cmd, self.u_bar[0].copy(), self.u_bar.copy()
    
    def evaluate_cost_function(self, cost_function: CostFunctions, x_bar, u_bar, nlp_p_map_bar):
        ps = []
        cost_p_dict = cost_function.get_p_dict()
        cost_p_struct = casadi_sym_struct(cost_p_dict)
        cost_p_map = cost_p_struct(0)

        vals = []
        for k in range(self.N):
            nlp_p_map = nlp_p_map_bar[k]
            for key in cost_p_map.keys():
                cost_p_map[key] = nlp_p_map[key]
            
            vals.append(cost_function.evaluate(x_bar[k], u_bar[k], cost_p_map.cat.full().flatten()))
        return np.sum(vals)
            