from abc import ABC, abstractmethod
import casadi as cs

import rospkg
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag

from mmseq_control.robot import MobileManipulator3D
from mmseq_utils.casadi import casadi_sym_struct

class RBF:
    mu_sym = cs.MX.sym('mu')
    zeta_sym = cs.MX.sym('zeta')
    h_sym = cs.MX.sym('h')

    B_eqn_list = [-mu_sym * cs.log(h_sym),
                  mu_sym * (0.5 * (((h_sym - 2 * zeta_sym) / zeta_sym) ** 2 - 1) - cs.log(zeta_sym))]
    s_eqn = h_sym < zeta_sym
    B_eqn = cs.conditional(s_eqn, B_eqn_list, 0, False)
    B_fcn = cs.Function("B_fcn", [h_sym, mu_sym, zeta_sym], [B_eqn])

    B_hess_eqn, B_grad_eqn = cs.hessian(B_eqn, h_sym)
    B_hess_fcn = cs.Function("ddBddh_fcn", [h_sym, mu_sym, zeta_sym], [B_hess_eqn])
    B_grad_fcn = cs.Function("dBdh_fcn", [h_sym, mu_sym, zeta_sym], [B_grad_eqn])


class CostFunctions(ABC):
    def __init__(self, nx: int, nu:int, name: str="MPCCost"):
        """ MPC cost functions base class
                        \mathcal{J}
        :param nx: state dim
        :param nu: control dim
        :param xsym: state, CasADi symbolic variable
        :param usym: control, CasADi symbolic variable
        :param psym: params, CasADi symbolic variable
        :param name: name, string

        """

        self.nx = nx
        self.nu = nu
        self.name = name

        self.x_sym = cs.MX.sym('x', nx)
        self.u_sym = cs.MX.sym('u', nu)

        self.p_sym = None
        self.p_dict = None
        self.p_struct = None
        self.J_eqn = None
        self.J_fcn = None

        super().__init__()
        
    def evaluate(self, x, u, p):
        if self.J_fcn is not None:
            return self.J_fcn(x,u,p).toarray()[0]
        else:
            return None

    def get_p_dict(self):
        if self.p_dict is None:
            return None
        else:
            return {key +f"_{self.name}":val for (key, val) in self.p_dict.items()}

class NonlinearLeastSquare(CostFunctions):
    def __init__(self, nx:int, nu:int, nr:int, f_fcn:cs.Function, W, name):
        """ Nonlinear least square cost function
            J = ||f_fcn(x) - r||^2_W

        :param dt: discretization time step
        :param nx: state dim
        :param nu: control dim
        :param N:  prediction window
        :param nr: ref dim
        :param f_fcn: output/observation function, casadi.Function
        :param params: cost function params
        """
        super().__init__(nx, nu, name)
        self.nr = nr
        self.f_fcn = f_fcn.expand()

        self.p_dict = {"r": cs.MX.sym("r_"+self.name, self.nr)}
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self.W = W
        self._setupSymMdl()

    def _setupSymMdl(self):
        y = self.f_fcn(self.x_sym)
        e = y - self.p_sym
        self.J_eqn = 0.5 * e.T @ self.W @ e
        self.J_fcn = cs.Function("J_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn], ["x", "u", "r"], ["J"]).expand()


class TrajectoryTrackingCostFunction(NonlinearLeastSquare):
    def __init__(self, nx: int, nu: int, nr: int, f_fcn: cs.Function, W, name):
        super().__init__(nx, nu, nr, f_fcn, W, name)
        self.e_eqn = self.f_fcn(self.x_sym) - self.p_sym
        self.e_fcn = cs.Function("e_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.e_eqn], ["x", "u", "r"], ["e"]).expand()


class EEPos3CostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 3
        fk_ee = robot_mdl.kinSymMdls[robot_mdl.tool_link_name]
        f_fcn = cs.Function("fee", [robot_mdl.x_sym], [fk_ee(robot_mdl.q_sym)[0]])
        super().__init__(nx, nu, nr, f_fcn, params["Qk"], "EEPos3")

class EEPos3BaseFrameCostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 3
        fk_ee = robot_mdl._getFk(robot_mdl.tool_link_name, base_frame=True)
        f_fcn = cs.Function("fee", [robot_mdl.x_sym], [fk_ee(robot_mdl.q_sym)[0]])
        super().__init__(nx, nu, nr, f_fcn, params["Qk"], "EEPos3BaseFrame")

class ArmJointCostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):

        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 6
        f_fcn = cs.Function("f_qa", [robot_mdl.x_sym], [robot_mdl.qa_sym])

        super().__init__(nx, nu, nr, f_fcn, params["Qk"], "ArmJoint")


class BasePos2CostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 2
        fk_b = robot_mdl.kinSymMdls["base"]
        f_fcn = cs.Function("fb", [robot_mdl.x_sym], [fk_b(robot_mdl.q_sym)[0]])

        super().__init__(nx, nu, nr, f_fcn, params["Qk"], "BasePos2")

class ControlEffortCostFunction(CostFunctions):
    def __init__(self, robot_mdl, params):

        ss_mdl = robot_mdl.ssSymMdl
        self.params = params
        super().__init__(ss_mdl["nx"], ss_mdl["nu"], "ControlEffort")

        self.p_dict = {}
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self._setupSymMdl()


    def _setupSymMdl(self):
        Qq = [self.params["Qqb"]]*3+ [self.params["Qqa"]]*6 + [self.params["Qvb"]]*3 + [self.params["Qva"]]*6
        Qx = np.diag(Qq)

        Qu = [self.params["Qub"]]*3+ [self.params["Qua"]]*6
        Qu = np.diag(Qu )

        self.J_eqn = 0.5 * self.x_sym.T @ Qx @ self.x_sym + 0.5 * self.u_sym.T @ Qu @ self.u_sym
        self.J_fcn = cs.Function("J_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn], ["x", "u", "r"], ["J"]).expand()
    
class SoftConstraintsRBFCostFunction(CostFunctions):
    def __init__(self, mu, zeta, cst_obj, name="SoftConstraint", expand=True):
        super().__init__(cst_obj.nx, cst_obj.nu, name)

        self.mu = mu
        self.zeta = zeta

        self.p_dict = cst_obj.get_p_dict()
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self.h_eqn = -cst_obj.g_fcn(self.x_sym, self.u_sym, self.p_sym)
        self.nh = self.h_eqn.shape[0]

        J_eqn_list = [RBF.B_fcn(self.h_eqn[k], self.mu, self.zeta) for k in range(self.nh)]
        self.J_eqn = sum(J_eqn_list)
        self.J_vec_eqn = cs.vertcat(*J_eqn_list)

        self.h_fcn = cs.Function("h_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.h_eqn])
        self.J_fcn = cs.Function("J_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn])
        self.J_vec_fcn = cs.Function("J_vec_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_vec_eqn])
            
        if expand:
            self.h_fcn = self.h_fcn.expand()
            self.J_fcn = self.J_fcn.expand()
            self.J_vec_fcn = self.J_vec_fcn.expand()
        
    def evaluate_vec(self, x, u, p):
        return self.J_vec_fcn(x, u, p).toarray().flatten()
    
    def get_p_dict(self):
        return self.p_dict