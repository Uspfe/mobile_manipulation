import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Tuple

import casadi as cs
import mobile_manipulation_central as mm
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from numpy.typing import NDArray
from scipy.interpolate import interp1d

import mm_control.MPCConstraints as MPCConstraints
from mm_control.MPCConstraints import ControlBoxConstraints, StateBoxConstraints
from mm_control.MPCCostFunctions import (
    CostFunctionRegistry,
    CostFunctions,
    SoftConstraintsRBFCostFunction,
)
from mm_control.robot import CasadiModelInterface as ModelInterface
from mm_control.robot import MobileManipulator3D as MM
from mm_utils.casadi_struct import casadi_sym_struct
from mm_utils.math import wrap_pi_array
from mm_utils.parsing import parse_ros_path

INF = 1e5
BASE_Z_HEIGHT_FOR_SDF = 0.2  # Base z height for SDF queries (meters)


class MPCBase:
    """Base class for Model Predictive Control"""

    def __init__(self, config):
        self.model_interface = ModelInterface(config)
        self.robot = self.model_interface.robot
        self.ssSymMdl = self.robot.ssSymMdl
        self.kinSymMdl = self.robot.kinSymMdls
        self.nx = self.ssSymMdl["nx"]
        self.nu = self.ssSymMdl["nu"]
        self.DoF = self.robot.DoF
        self.home = mm.load_home_position(config.get("home", "default"))

        self.params = config
        self.dt = self.params["dt"]
        self.tf = self.params["prediction_horizon"]
        self.N = int(self.tf / self.dt)
        self.QPsize = self.nx * (self.N + 1) + self.nu * self.N

        self.collision_link_names = (
            ["self"] if self.params["self_collision_avoidance_enabled"] else []
        )
        self.collision_link_names += (
            self.model_interface.scene.collision_link_names["static_obstacles"]
            if self.params["static_obstacles_collision_avoidance_enabled"]
            else []
        )
        self.collision_link_names += (
            ["sdf"] if self.params["sdf_collision_avoidance_enabled"] else []
        )

        self.collisionCsts = {}
        for name in self.collision_link_names:
            self.collisionCsts[name] = self._create_collision_constraint(name)

        self.collisionSoftCsts = {}
        for name, sd_cst in self.collisionCsts.items():
            self.collisionSoftCsts[name] = self._create_collision_soft_cost(
                name, sd_cst
            )

        self.stateCst = StateBoxConstraints(self.robot)
        self.controlCst = ControlBoxConstraints(self.robot)

        self.cost = []
        self.constraints = []

        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.x_bar[:, : self.DoF] = self.home
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.t_bar = None
        self.lam_bar = None  # inequality multipliers
        self.u_prev = np.zeros(self.nu)

        self.v_cmd = np.zeros(self.nx - self.DoF)

        self.py_logger = logging.getLogger("Controller")
        self.log = self._get_log()

        self.ree_bar = None
        self.rbase_bar = None
        self.ee_bar = None
        self.base_bar = None
        self.sdf_bar = {"EE": None, "base": None}
        self.sdf_grad_bar = {"EE": None, "base": None}

        self.output_dir = Path(
            parse_ros_path({"package": "mm_control", "path": "acados_outputs"})
        )
        if not self.output_dir.exists():
            self.output_dir.mkdir()

    @abstractmethod
    def control(
        self,
        t: float,
        robot_states: Tuple[NDArray[np.float64], NDArray[np.float64]],
        references: dict,
        map=None,
    ):
        """
        :param t: current control time
        :param robot_states: (q, v) generalized coordinates and velocities
        :param references: Dictionary with reference trajectories from TaskManager:
            {
                "base_pose": array of shape (N+1, 3) or None,
                "base_velocity": array of shape (N+1, 3) or None,
                "ee_pose": array of shape (N+1, 6) or None,
                "ee_velocity": array of shape (N+1, 6) or None,
            }
        :param map: SDF map data (optional)
        :return: u, currently the best control inputs, aka, u_bar[0]
        """
        pass

    def _predictTrajectories(self, xo, u_bar):
        return MM.ssIntegrate(self.dt, xo, u_bar, self.ssSymMdl)

    def _getEEBaseTrajectories(self, x_bar):
        ee_bar = np.zeros((self.N + 1, 3))
        base_bar = np.zeros((self.N + 1, 3))
        for k in range(self.N + 1):
            base_bar[k] = x_bar[k, :3]
            fee_fcn = self.kinSymMdl[self.robot.tool_link_name]
            ee_pos, ee_orn = fee_fcn(x_bar[k, : self.DoF])
            ee_bar[k] = ee_pos.toarray().flatten()

        return ee_bar, base_bar

    def reset(self):
        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.t_bar = None
        self.lam_bar = None

        self.v_cmd = np.zeros(self.nx - self.DoF)

    def evaluate_cost_function(
        self, cost_function: CostFunctions, x_bar, u_bar, nlp_p_map_bar
    ):
        cost_p_dict = cost_function.get_p_dict()
        cost_p_struct = casadi_sym_struct(cost_p_dict)
        cost_p_map = cost_p_struct(0)

        vals = []
        for k in range(self.N):
            nlp_p_map = nlp_p_map_bar[k]
            for key in cost_p_map.keys():
                cost_p_map[key] = nlp_p_map[key]
            v = cost_function.evaluate(
                x_bar[k], u_bar[k], cost_p_map.cat.full().flatten()
            )
            vals.append(v)
        return np.sum(vals)

    def evaluate_constraints(
        self, constraints: MPCConstraints.Constraint, x_bar, u_bar, nlp_p_map_bar
    ):
        cost_p_dict = constraints.get_p_dict()
        cost_p_struct = casadi_sym_struct(cost_p_dict)
        cost_p_map = cost_p_struct(0)

        vals = []
        for k in range(self.N + 1):
            nlp_p_map = nlp_p_map_bar[k]
            for key in cost_p_map.keys():
                cost_p_map[key] = nlp_p_map[key]
            if k < self.N:
                v = constraints.check(
                    x_bar[k], u_bar[k], cost_p_map.cat.full().flatten()
                )
            else:
                v = constraints.check(
                    x_bar[k], u_bar[k - 1], cost_p_map.cat.full().flatten()
                )

            vals.append(v)
        return vals

    def _construct(self, costs, constraints, num_terminal_cost, name="MM"):
        model, p_struct, p_map = self._setup_acados_model(costs, constraints, name)
        ocp = self._setup_acados_ocp(model, name)
        self._setup_costs(ocp, model, costs, num_terminal_cost)
        nsx, nsu, nsx_e, nsh, nsh_e, nsh_0 = self._setup_constraints(
            ocp, model, constraints
        )
        self._setup_slack_variables(ocp, nsx, nsu, nsh, nsx_e, nsh_e, nsh_0)

        ocp.constraints.x0 = self.x_bar[0]
        ocp.parameter_values = p_map.cat.full().flatten()
        self._configure_solver_options(ocp)
        ocp_solver = self._create_solver(ocp, name)

        return ocp, ocp_solver, p_struct

    def _setup_acados_model(self, costs, constraints, name):
        """Setup AcadosModel with dynamics and parameters."""
        model = AcadosModel()
        model.x = cs.MX.sym("x", self.nx)
        model.u = cs.MX.sym("u", self.nu)
        model.xdot = cs.MX.sym("xdot", self.nx)

        model.f_impl_expr = model.xdot - self.ssSymMdl["fmdl"](model.x, model.u)
        model.f_expl_expr = self.ssSymMdl["fmdl"](model.x, model.u)
        model.name = name

        # Get params from costs and constraints
        p_dict = {}
        for cost in costs:
            p_dict.update(cost.get_p_dict())
        for cst in constraints:
            p_dict.update(cst.get_p_dict())
        p_struct = casadi_sym_struct(p_dict)
        p_map = p_struct(0)
        model.p = p_struct.cat

        return model, p_struct, p_map

    def _setup_acados_ocp(self, model, name):
        """Setup AcadosOCP basic structure."""
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.tf
        ocp.code_export_directory = str(self.output_dir / "c_generated_code")
        ocp.solver_options.ext_fun_compile_flags = "-O3"
        return ocp

    def _setup_costs(self, ocp, model, costs, num_terminal_cost):
        """Setup cost expressions for the OCP."""
        ocp.cost.cost_type = "EXTERNAL"
        cost_expr = []
        for cost in costs:
            Ji = cost.J_fcn(model.x, model.u, cost.p_sym)
            cost_expr.append(Ji)
        ocp.model.cost_expr_ext_cost = sum(cost_expr)

        custom_hess_expr = []
        if self.params["acados"]["use_custom_hess"]:
            for cost in costs:
                H_fcn = cost.get_custom_H_fcn()
                H_expr_i = H_fcn(model.x, model.u, cost.p_sym)
                custom_hess_expr.append(H_expr_i)
            ocp.model.cost_expr_ext_cost_custom_hess = sum(custom_hess_expr)

        if self.params["acados"]["use_terminal_cost"]:
            ocp.cost.cost_type_e = "EXTERNAL"
            cost_expr_e = sum(cost_expr[:num_terminal_cost])
            cost_expr_e = cs.substitute(cost_expr_e, model.u, [])
            model.cost_expr_ext_cost_e = cost_expr_e
            if self.params["acados"]["use_custom_hess"]:
                cost_hess_expr_e = sum(custom_hess_expr[:num_terminal_cost])
                cost_hess_expr_e = cs.substitute(cost_hess_expr_e, model.u, [])
                model.cost_expr_ext_cost_custom_hess_e = cost_hess_expr_e

    def _setup_constraints(self, ocp, model, constraints):
        """Setup all constraints (control, state, nonlinear) and return slack variable counts."""
        # Control input constraints
        ocp.constraints.lbu = np.array(self.ssSymMdl["lb_u"])
        ocp.constraints.ubu = np.array(self.ssSymMdl["ub_u"])
        ocp.constraints.idxbu = np.arange(self.nu)

        if self.params["acados"]["slack_enabled"]["u"]:
            ocp.constraints.idxsbu = np.arange(self.nu)
            ocp.constraints.lsbu = np.zeros(self.nu)
            ocp.constraints.usbu = np.zeros(self.nu)
            nsu = self.nu
        else:
            nsu = 0

        # State constraints
        ocp.constraints.lbx = np.array(self.ssSymMdl["lb_x"])
        ocp.constraints.ubx = np.array(self.ssSymMdl["ub_x"])
        ocp.constraints.idxbx = np.arange(self.nx)

        if self.params["acados"]["slack_enabled"]["x"]:
            ocp.constraints.idxsbx = np.arange(self.nx)
            ocp.constraints.lsbx = np.zeros(self.nx)
            ocp.constraints.usbx = np.zeros(self.nx)
            nsx = self.nx
        else:
            nsx = 0

        ocp.constraints.lbx_e = np.array(self.ssSymMdl["lb_x"])
        ocp.constraints.ubx_e = np.array(self.ssSymMdl["ub_x"])
        ocp.constraints.idxbx_e = np.arange(self.nx)

        if self.params["acados"]["slack_enabled"]["x_e"]:
            ocp.constraints.idxsbx_e = np.arange(self.nx)
            ocp.constraints.lsbx_e = np.zeros(self.nx)
            ocp.constraints.usbx_e = np.zeros(self.nx)
            nsx_e = self.nx
        else:
            nsx_e = 0

        # Nonlinear constraints
        h_expr_list = []
        idxsh = []
        h_idx = 0
        for cst in constraints:
            h_expr_list.append(cst.g_fcn(model.x, model.u, cst.p_sym))
            if cst.slack_enabled and (
                self.params["acados"]["slack_enabled"]["h"]
                or self.params["acados"]["slack_enabled"]["h_0"]
                or self.params["acados"]["slack_enabled"]["h_e"]
            ):
                idxsh += [h_i for h_i in range(h_idx, h_idx + cst.ng)]
            h_idx += cst.ng

        nsh = len(idxsh) if self.params["acados"]["slack_enabled"]["h"] else 0
        nsh_e = len(idxsh) if self.params["acados"]["slack_enabled"]["h_e"] else 0
        nsh_0 = len(idxsh) if self.params["acados"]["slack_enabled"]["h_0"] else 0

        if len(h_expr_list) > 0:
            h_expr = cs.vertcat(*h_expr_list)
            h_expr_num = h_expr.shape[0]

            model.con_h_expr_0 = h_expr
            ocp.constraints.uh_0 = np.zeros(h_expr_num)
            ocp.constraints.lh_0 = -INF * np.ones(h_expr_num)

            model.con_h_expr = h_expr
            ocp.constraints.uh = np.zeros(h_expr_num)
            ocp.constraints.lh = -INF * np.ones(h_expr_num)

            model.con_h_expr_e = cs.substitute(h_expr, model.u, [])
            ocp.constraints.uh_e = np.zeros(h_expr_num)
            ocp.constraints.lh_e = -INF * np.ones(h_expr_num)

            if nsh_0 > 0:
                ocp.constraints.idxsh_0 = np.array(idxsh)
                ocp.constraints.lsh_0 = np.zeros(nsh_0)
                ocp.constraints.ush_0 = np.zeros(nsh_0)
            if nsh > 0:
                ocp.constraints.idxsh = np.array(idxsh)
                ocp.constraints.lsh = np.zeros(nsh)
                ocp.constraints.ush = np.zeros(nsh)
            if nsh_e > 0:
                ocp.constraints.idxsh_e = np.array(idxsh)
                ocp.constraints.lsh_e = np.zeros(nsh_e)
                ocp.constraints.ush_e = np.zeros(nsh_e)

        return nsx, nsu, nsx_e, nsh, nsh_e, nsh_0

    def _configure_solver_options(self, ocp):
        """Configure solver options from config."""
        for key, val in self.params["acados"]["ocp_solver_options"].items():
            attr = getattr(ocp.solver_options, key, None)
            if attr is not None:
                setattr(ocp.solver_options, key, val)
            else:
                self.py_logger.warning(
                    f"{key} not found in Acados solver options. Parameter is ignored."
                )

    def _create_solver(self, ocp, name):
        """Create and return AcadosOCPSolver."""
        json_file_name = str(self.output_dir / f"acados_ocp_{name}.json")
        if self.params["acados"]["cython"]["enabled"]:
            if self.params["acados"]["cython"]["recompile"]:
                AcadosOcpSolver.generate(ocp, json_file=json_file_name)
                AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
                return AcadosOcpSolver.create_cython_solver(json_file_name)
            else:
                return AcadosOcpSolver(
                    ocp, json_file=json_file_name, build=False, generate=False
                )
        else:
            return AcadosOcpSolver(ocp, json_file=json_file_name, build=True)

    def _create_collision_constraint(self, name):
        """Create a collision constraint for the given link name."""
        sd_fcn = self.model_interface.getSignedDistanceSymMdls(name)
        is_static = (
            name in self.model_interface.scene.collision_link_names["static_obstacles"]
        )

        if is_static:
            constraint_type_name = self.params["collision_constraint_type"][
                "static_obstacles"
            ]
            safety_margin = self.params["collision_safety_margin"]["static_obstacles"]
        else:
            constraint_type_name = self.params["collision_constraint_type"][name]
            safety_margin = self.params["collision_safety_margin"][name]

        collision_cst_type = getattr(MPCConstraints, constraint_type_name)
        return collision_cst_type(self.robot, sd_fcn, safety_margin, name)

    def _create_collision_soft_cost(self, name, sd_cst):
        """Create a soft collision cost for the given constraint."""
        expand = name != "sdf"
        is_static = (
            name in self.model_interface.scene.collision_link_names["static_obstacles"]
        )

        if is_static:
            mu = self.params["collision_soft"]["static_obstacles"]["mu"]
            zeta = self.params["collision_soft"]["static_obstacles"]["zeta"]
        else:
            mu = self.params["collision_soft"][name]["mu"]
            zeta = self.params["collision_soft"][name]["zeta"]

        return SoftConstraintsRBFCostFunction(
            mu, zeta, sd_cst, name + "CollisionSoftCst", expand=expand
        )

    def _setup_slack_variables(self, ocp, nsx, nsu, nsh, nsx_e, nsh_e, nsh_0):
        """Setup slack variables for the OCP."""
        z = self.params["cost_params"]["slack"]["z"]
        Z = self.params["cost_params"]["slack"]["Z"]

        ns = nsx + nsu + nsh
        if ns > 0:
            ocp.cost.Zl = np.ones(ns) * Z
            ocp.cost.Zu = np.ones(ns) * Z
            ocp.cost.zl = np.ones(ns) * z
            ocp.cost.zu = np.ones(ns) * z

        ns_e = nsx_e + nsh_e
        if ns_e > 0:
            ocp.cost.Zl_e = np.ones(ns_e) * Z
            ocp.cost.Zu_e = np.ones(ns_e) * Z
            ocp.cost.zl_e = np.ones(ns_e) * z
            ocp.cost.zu_e = np.ones(ns_e) * z

        ns_0 = nsh_0 + nsu
        if ns_0 > 0:
            ocp.cost.Zl_0 = np.ones(ns_0) * Z
            ocp.cost.Zu_0 = np.ones(ns_0) * Z
            ocp.cost.zl_0 = np.ones(ns_0) * z
            ocp.cost.zu_0 = np.ones(ns_0) * z

    def _get_log(self):
        return {}


class MPC(MPCBase):
    """Single-task Model Predictive Controller using Acados"""

    def __init__(self, config):
        super().__init__(config)
        num_terminal_cost = 2
        cost_params = config["cost_params"]

        # Create cost functions using simplified parameterized registry
        costs = []

        # Base costs - always use SE2 (yaw tracking controlled via weights, set yaw weight to 0 to disable)
        costs.append(
            CostFunctionRegistry.create(
                "BasePose", self.robot, cost_params.get("BasePose", {}), dimension="SE2"
            )
        )
        costs.append(
            CostFunctionRegistry.create(
                "BaseVel", self.robot, cost_params.get("BaseVel", {}), dimension=3
            )
        )

        # EE costs - always use SE3 (orientation weights can be set to 0 if not needed)
        costs.append(
            CostFunctionRegistry.create(
                "EEPose",
                self.robot,
                cost_params.get("EEPose", {}),
                pose_type="SE3",
                frame="world",
            )
        )
        costs.append(
            CostFunctionRegistry.create(
                "EEVel", self.robot, cost_params.get("EEVel", {})
            )
        )

        # Control effort (always included)
        costs.append(
            CostFunctionRegistry.create(
                "ControlEffort", self.robot, cost_params.get("Effort", {})
            )
        )

        # Add collision costs/constraints
        constraints = []
        for name in self.collision_link_names:
            # fmt: off
            is_static = name in self.model_interface.scene.collision_link_names["static_obstacles"]
            softened = self.params["collision_constraints_softened"]["static_obstacles" if is_static else name]
            # fmt: on
            if softened:
                costs.append(self.collisionSoftCsts[name])
            else:
                constraints.append(self.collisionCsts[name])

        name = self.params["acados"].get("name", "MM")
        self.ocp, self.ocp_solver, self.p_struct = self._construct(
            costs, constraints, num_terminal_cost, name
        )

        self.cost = costs
        self.constraints = constraints + [self.controlCst, self.stateCst]

    def _get_config_key_for_cost_name(self, cost_name):
        """Map cost function name to simplified config parameter key."""
        name_mapping = {
            "EEPoseSE3": "EEPose",
            "EEVel6": "EEVel",
            "BasePoseSE2": "BasePose",
            "BaseVel3": "BaseVel",
        }
        return name_mapping.get(cost_name, cost_name)

    def _set_control_effort_params(self, curr_p_map):
        """Set ControlEffort cost function parameters in the parameter map."""
        effort_params = self.params["cost_params"]["Effort"]
        for param_name in ["Qqa", "Qqb", "Qva", "Qvb", "Qua", "Qub"]:
            curr_p_map[f"{param_name}_ControlEffort"] = effort_params[param_name]

    def control(
        self,
        t: float,
        robot_states: Tuple[NDArray[np.float64], NDArray[np.float64]],
        references: dict,
        map=None,
    ):
        self.py_logger.debug("control time {}".format(t))
        self.curr_control_time = t
        q, v = robot_states
        q[2:9] = wrap_pi_array(q[2:9])
        xo = np.hstack((q, v))

        x_bar_initial, u_bar_initial = self._prepare_warm_start(t, xo)
        r_bar_map = self._convert_references_to_r_bar_map(references, xo)
        self._update_sdf_map(map)
        curr_p_map_bar = self._setup_horizon_parameters(
            r_bar_map, x_bar_initial, u_bar_initial
        )
        self._solve_and_extract(xo, t, curr_p_map_bar, x_bar_initial, u_bar_initial)
        self._update_logging(curr_p_map_bar)

        return (
            self.v_cmd,
            self.u_prev,
            self.u_bar.copy(),
            self.x_bar[:, self.DoF :].copy(),
        )

    def _prepare_warm_start(self, t, xo):
        """Prepare warm start trajectories from previous solution or zeros."""
        if self.t_bar is not None:
            self.u_t = interp1d(
                self.t_bar,
                self.u_bar,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )
            t_bar_new = t + np.arange(self.N) * self.dt
            self.u_bar = self.u_t(t_bar_new)
            self.x_bar = self._predictTrajectories(xo, self.u_bar)
        else:
            self.u_bar = np.zeros_like(self.u_bar)
            self.x_bar = self._predictTrajectories(xo, self.u_bar)

        return self.x_bar.copy(), self.u_bar.copy()

    def _convert_references_to_r_bar_map(self, references, xo):
        """Convert references from TaskManager format to MPC cost function format.

        Args:
            references: Dictionary from TaskManager with keys:
                - "base_pose": array of shape (N+1, 3) or None
                - "base_velocity": array of shape (N+1, 3) or None
                - "ee_pose": array of shape (N+1, 6) or None
                - "ee_velocity": array of shape (N+1, 6) or None
            xo: Current state vector [q, v] to compute current EE pose if needed

        Returns:
            Dictionary with cost function names as keys:
                - "BasePoseSE2": list of arrays (N+1, 3)
                - "BaseVel3": list of arrays (N+1, 3)
                - "EEPoseSE3": list of arrays (N+1, 6)
                - "EEVel6": list of arrays (N+1, 6)
        """
        r_bar_map = {}

        # Convert base pose reference - only if provided
        if references.get("base_pose") is not None:
            base_pose = references["base_pose"]
            r_bar_map["BasePoseSE2"] = [base_pose[i] for i in range(self.N + 1)]
            self.rbase_bar = (
                base_pose.tolist() if hasattr(base_pose, "tolist") else base_pose
            )
        else:
            # No base reference: set empty list for visualization
            self.rbase_bar = []

        # Convert base velocity reference - only if provided
        if references.get("base_velocity") is not None:
            base_vel = references["base_velocity"]
            r_bar_map["BaseVel3"] = [base_vel[i] for i in range(self.N + 1)]

        # Convert EE pose reference - only if provided
        if references.get("ee_pose") is not None:
            ee_pose = references["ee_pose"]
            # EE reference is in world frame
            r_bar_map["EEPoseSE3"] = [ee_pose[i] for i in range(self.N + 1)]
            self.ree_bar = ee_pose.tolist() if hasattr(ee_pose, "tolist") else ee_pose
        else:
            # No EE reference: set empty list for visualization
            self.ree_bar = []

        # Convert EE velocity reference - only if provided
        if references.get("ee_velocity") is not None:
            ee_vel = references["ee_velocity"]
            r_bar_map["EEVel6"] = [ee_vel[i] for i in range(self.N + 1)]

        return r_bar_map

    def _quat_to_rpy(self, quat):
        """Convert quaternion to Euler angles [roll, pitch, yaw] (xyz order).

        Args:
            quat: Quaternion in [x, y, z, w] format (from spatialmath)

        Returns:
            Euler angles [roll, pitch, yaw] in xyz order
        """
        from scipy.spatial.transform import Rotation as Rot

        # spatialmath returns [x, y, z, w], scipy expects [x, y, z, w]
        r = Rot.from_quat(quat)
        return r.as_euler("xyz")

    def _update_sdf_map(self, map):
        """Update SDF map if provided and enabled."""
        t1 = time.perf_counter()
        if map is not None and self.params["sdf_collision_avoidance_enabled"]:
            self.model_interface.sdf_map.update_map(*map)
        t2 = time.perf_counter()
        self.log["time_map_update"] = t2 - t1

    def _setup_horizon_parameters(self, r_bar_map, x_bar_initial, u_bar_initial):
        """Setup OCP parameters for each horizon step."""
        tp1 = time.perf_counter()
        curr_p_map_bar = []

        # Reset time logging
        for key in self.log.keys():
            if "time" in key:
                self.log[key] = 0

        map_params = self.model_interface.sdf_map.get_params()
        for i in range(self.N + 1):
            curr_p_map = self.p_struct(0)
            self._set_sdf_params(curr_p_map, map_params)
            self._set_cbf_gamma_params(curr_p_map)
            self._set_initial_guess(curr_p_map, i, x_bar_initial, u_bar_initial)
            self._set_tracking_params(curr_p_map, r_bar_map, i)
            self._set_control_effort_params(curr_p_map)
            self._set_ocp_params(curr_p_map, i)
            curr_p_map_bar.append(curr_p_map)

        tp2 = time.perf_counter()
        self.log["time_ocp_set_params"] = tp2 - tp1
        return curr_p_map_bar

    def _set_sdf_params(self, curr_p_map, map_params):
        """Set SDF map parameters in the parameter map."""
        t1 = time.perf_counter()
        if self.params["sdf_collision_avoidance_enabled"]:
            curr_p_map["x_grid_sdf"] = map_params[0]
            curr_p_map["y_grid_sdf"] = map_params[1]
            if self.model_interface.sdf_map.dim == 3:
                curr_p_map["z_grid_sdf"] = map_params[2]
                curr_p_map["value_sdf"] = map_params[3]
            else:
                curr_p_map["value_sdf"] = map_params[2]
        t2 = time.perf_counter()
        self.log["time_ocp_set_params_map"] += t2 - t1

    def _set_cbf_gamma_params(self, curr_p_map):
        """Set CBF gamma parameters for collision constraints."""
        for name in self.collision_link_names:
            is_static = (
                name
                in self.model_interface.scene.collision_link_names["static_obstacles"]
            )
            constraint_type = (
                self.params["collision_constraint_type"]["static_obstacles"]
                if is_static
                else self.params["collision_constraint_type"][name]
            )

            if constraint_type == "SignedDistanceConstraintCBF":
                p_name = "_".join(["gamma", name])
                curr_p_map[p_name] = self.params["collision_cbf_gamma"][name]

    def _set_initial_guess(self, curr_p_map, i, x_bar_initial, u_bar_initial):
        """Set initial guess for state, control, and multipliers."""
        t1 = time.perf_counter()
        self.ocp_solver.set(i, "x", x_bar_initial[i])
        if i < self.N:
            self.ocp_solver.set(i, "u", u_bar_initial[i])
        if self.lam_bar is not None:
            self.ocp_solver.set(i, "lam", self.lam_bar[i])
        t2 = time.perf_counter()
        self.log["time_ocp_set_params_set_x"] += t2 - t1

    def _set_tracking_params(self, curr_p_map, r_bar_map, i):
        """Set tracking cost function parameters."""
        t1 = time.perf_counter()
        p_keys = self.p_struct.keys()

        # List of all possible tracking cost functions (world frame only)
        all_tracking_costs = ["BasePoseSE2", "BaseVel3", "EEPoseSE3", "EEVel6"]

        for name in all_tracking_costs:
            p_name_r = f"r_{name}"  # Reference parameter for the tracking cost (e.g., r_EEPoseSE3)
            p_name_W = f"W_{name}"  # Weight matrix parameter for the tracking cost (e.g., W_EEPoseSE3)

            if p_name_r in p_keys:
                if name in r_bar_map:
                    # Reference provided: set reference and use configured weights
                    curr_p_map[p_name_r] = r_bar_map[name][i]
                    config_key = self._get_config_key_for_cost_name(name)
                    cost_params = self.params["cost_params"].get(config_key, {})
                    weight_key = "P" if i == self.N else "Qk"
                    weights = cost_params.get(
                        weight_key, [1.0] * len(r_bar_map[name][i])
                    )
                    curr_p_map[p_name_W] = np.diag(weights)
                else:
                    # No reference provided: set weights to zero (minimize control effort only)
                    config_key = self._get_config_key_for_cost_name(name)
                    cost_params = self.params["cost_params"].get(config_key, {})
                    weight_key = "P" if i == self.N else "Qk"
                    # Get dimension from default weights
                    default_weights = cost_params.get(weight_key, [1.0])
                    if isinstance(default_weights, (list, np.ndarray)):
                        dim = len(default_weights)
                    else:
                        # Scalar weight - determine dimension from cost function name
                        if name == "EEPoseSE3":
                            dim = 6
                        elif name == "BasePoseSE2":
                            dim = 3
                        elif name == "EEVel6":
                            dim = 6
                        elif name == "BaseVel3":
                            dim = 3
                        else:
                            raise ValueError(f"Unknown cost function name: {name}")
                    # Set zero reference and zero weights
                    curr_p_map[p_name_r] = np.zeros(dim)
                    curr_p_map[p_name_W] = np.diag(np.zeros(dim))
            else:
                raise RuntimeError(f"Parameter {p_name_r} not found in p_struct keys")

        t2 = time.perf_counter()
        self.log["time_ocp_set_params_tracking"] += t2 - t1

    def _set_ocp_params(self, curr_p_map, i):
        """Set OCP parameters for the current horizon step."""
        t1 = time.perf_counter()
        self.ocp_solver.set(i, "p", curr_p_map.cat.full().flatten())
        t2 = time.perf_counter()
        self.log["time_ocp_set_params_setp"] += t2 - t1

    def _solve_and_extract(self, xo, t, curr_p_map_bar, x_bar_initial, u_bar_initial):
        """Solve the OCP and extract solution."""
        t1 = time.perf_counter()
        self.ocp_solver.solve_for_x0(xo, fail_on_nonzero_status=False)
        t2 = time.perf_counter()
        self.log["time_ocp_solve"] = t2 - t1

        self.ocp_solver.print_statistics()
        self.log["solver_status"] = self.ocp_solver.status
        if self.ocp.solver_options.nlp_solver_type != "SQP_RTI":
            self.log["step_size"] = np.mean(self.ocp_solver.get_stats("alpha"))
        else:
            self.log["step_size"] = -1
        self.log["sqp_iter"] = self.ocp_solver.get_stats("sqp_iter")
        self.log["qp_iter"] = sum(self.ocp_solver.get_stats("qp_iter"))
        self.log["cost_final"] = self.ocp_solver.get_cost()

        if self.ocp_solver.status != 0:
            x_bar = [self.ocp_solver.get(i, "x") for i in range(self.N)]
            u_bar = [self.ocp_solver.get(i, "u") for i in range(self.N)]
            x_bar.append(self.ocp_solver.get(self.N, "x"))

            self.log["iter_snapshot"] = {
                "t": t,
                "xo": xo,
                "p_map_bar": [p.cat.full().flatten() for p in curr_p_map_bar],
                "x_bar_init": x_bar_initial,
                "u_bar_init": u_bar_initial,
                "x_bar": x_bar,
                "u_bar": u_bar,
            }

            if self.params["acados"]["raise_exception_on_failure"]:
                raise Exception(
                    f"acados acados_ocp_solver returned status {self.ocp_solver.status}"
                )
        else:
            self.log["iter_snapshot"] = None

        # Extract solution
        self.u_prev = self.u_bar[0].copy()
        self.lam_bar = []
        for i in range(self.N):
            self.x_bar[i, :] = self.ocp_solver.get(i, "x")
            self.u_bar[i, :] = self.ocp_solver.get(i, "u")
            self.lam_bar.append(self.ocp_solver.get(i, "lam"))

        self.x_bar[self.N, :] = self.ocp_solver.get(self.N, "x")
        self.lam_bar.append(self.ocp_solver.get(self.N, "lam"))
        self.t_bar = t + np.arange(self.N) * self.dt
        self.v_cmd = self.x_bar[0][self.robot.DoF :].copy()

    def _update_logging(self, curr_p_map_bar):
        """Update logging and visualization data."""
        t1 = time.perf_counter()

        self.ee_bar, self.base_bar = self._getEEBaseTrajectories(self.x_bar)
        self.sdf_bar["EE"] = self.model_interface.sdf_map.query_val(
            self.ee_bar[:, 0], self.ee_bar[:, 1], self.ee_bar[:, 2]
        ).flatten()
        self.sdf_grad_bar["EE"] = self.model_interface.sdf_map.query_grad(
            self.ee_bar[:, 0], self.ee_bar[:, 1], self.ee_bar[:, 2]
        ).reshape((3, -1))

        base_z = np.ones(self.N + 1) * BASE_Z_HEIGHT_FOR_SDF
        self.sdf_bar["base"] = self.model_interface.sdf_map.query_val(
            self.base_bar[:, 0], self.base_bar[:, 1], base_z
        )
        self.sdf_grad_bar["base"] = self.model_interface.sdf_map.query_grad(
            self.base_bar[:, 0], self.base_bar[:, 1], base_z
        ).reshape((3, -1))

        for name in self.collision_link_names:
            self.log["_".join([name, "constraint"])] = self.evaluate_constraints(
                self.collisionCsts[name], self.x_bar, self.u_bar, curr_p_map_bar
            )

        self.log["ee_pos"] = self.ee_bar.copy()
        self.log["base_pos"] = self.base_bar.copy()
        self.log["ocp_param"] = [p.cat.full().flatten() for p in curr_p_map_bar]
        self.log["x_bar"] = self.x_bar.copy()
        self.log["u_bar"] = self.u_bar.copy()
        sdf_param = self.model_interface.sdf_map.get_params()
        for i, param in enumerate(sdf_param):
            self.log["_".join(["sdf", "param", str(i)])] = param
        t2 = time.perf_counter()
        self.log["time_ocp_overhead"] = t2 - t1

    def _get_log(self):
        log = {
            "cost_final": 0,
            "step_size": 0,
            "sqp_iter": 0,
            "qp_iter": 0,
            "solver_status": 0,
            "time_map_update": 0,
            "time_ocp_set_params": 0,
            "time_ocp_solve": 0,
            "time_ocp_set_params_map": 0,
            "time_ocp_set_params_set_x": 0,
            "time_ocp_set_params_tracking": 0,
            "time_ocp_set_params_setp": 0,
            "state_constraint": 0,
            "control_constraint": 0,
            "x_bar": 0,
            "u_bar": 0,
            "lam_bar": 0,
            "ee_pos": 0,
            "base_pos": 0,
            "ocp_param": {},
            "iter_snapshot": {},
        }
        for name in self.collision_link_names:
            log["_".join([name, "constraint"])] = 0
            log["_".join([name, "constraint", "gradient"])] = 0

        for i in range(self.model_interface.sdf_map.dim + 1):
            log["_".join(["sdf", "param", str(i)])] = 0
        return log

    def reset(self):
        super().reset()
        self.ocp_solver.reset()


if __name__ == "__main__":
    from mm_utils import parsing

    path_to_config = parse_ros_path(
        {"package": "mm_run", "path": "config/3d_collision_sdf.yaml"}
    )
    config = parsing.load_config(path_to_config)

    controller = MPC(config["controller"])
