#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:16:00 2023

@author: tracy
"""

from mmseq_control.robot import MobileManipulator3D as MM
from mmseq_control.IDKCTasks import EEPositionTracking, BasePositionTracking, JointVelocityBound, JointAngleBound
from mmseq_control.HQP import PrioritizedLinearSystemsNew as PLSN

from mmseq_utils import parsing

import numpy as np
import casadi as cs

from mmseq_control.HQP import PrioritizedLinearSystemsNew as PLSN
# from mmseq_control.HQP import PrioritizedLinearSystems as PLS
from qpsolvers import solve_qp, qpoases_solve_qp

class IDKC():

    def __init__(self, config):
        self.robot = MM(config)
        self.params = config
        self.QPsize = self.robot.DoF

        self.ee_pos_tracking = EEPositionTracking(self.robot, config)
        self.base_pos_tracking = BasePositionTracking(self.robot, config)
        self.qdot_bound = JointVelocityBound(self.robot, config)

    def control(self, t, robot_states, planners):
        q, _ = robot_states
        planner = planners[0]

        rd, vd = planner.getTrackingPoint(t, robot_states)

        if planner.type == "EE" and planner.ref_data_type == "Vec3":
                J, ed = self.ee_pos_tracking.linearize(q, rd, vd)
        elif planner.type == "base" and planner.ref_data_type == "Vec2":
                J, ed = self.base_pos_tracking.linearize(q, rd, vd)
        else:
            print("Planner of Type {} and Data type {} Not supported".format(planner.type, planner.ref_data_type))
            J = np.eye(self.robot.DoF)
            ed = np.zeros(self.robot.DoF)

        if self.params["solver"] == "pinv":
            qdot_d = (np.linalg.pinv(J) @ ed).toarray().flatten()
        elif self.params["solver"] == "QP":
            H = J.T @ J + self.params["ρ"] * cs.DM.eye(self.QPsize)
            g = - J.T @ ed
            qp = {}
            qp['h'] = H.sparsity()
            opts = {"error_on_fail": True,
                    "gurobi": {"OutputFlag": 0, "LogToConsole": 0, "Presolve": 1, "BarConvTol": 1e-8,
                               "OptimalityTol": 1e-6}}
            S = cs.conic('S', 'gurobi', qp, opts)

            results = S(h=H, g=g, lbx=-self.qdot_bound.ub[self.QPsize:], ubx=self.qdot_bound.ub[:self.QPsize])
            qdot_d = np.array(results['x']).squeeze()
        else:
            qdot_d = np.zeros(self.robot.DoF)
        return qdot_d, np.zeros(self.robot.DoF)

class HTIDKC():

    def __init__(self, config):
        self.robot = MM(config)
        self.params = config
        self.QPsize = self.robot.DoF

        self.ee_pos_tracking = EEPositionTracking(self.robot, config)
        self.base_pos_tracking = BasePositionTracking(self.robot, config)
        self.qdot_bound = JointVelocityBound(self.robot, config)
        self.q_bound = JointAngleBound(self.robot, config)

    def control(self, t, robot_states, planners):
        q, _ = robot_states
        Js = []
        eds = []
        task_types = []

        # joint angle bound
        Jq, edq = self.q_bound.linearize(q, [])
        Js.append(Jq)
        eds.append(edq)
        task_types.append("Ineq")

        for pid, planner in enumerate(planners):
            rd, vd = planner.getTrackingPoint(t, robot_states)

            if planner.type == "EE" and planner.ref_data_type == "Vec3":
                    J, ed = self.ee_pos_tracking.linearize(q, rd, vd)
            elif planner.type == "base" and planner.ref_data_type == "Vec2":
                    J, ed = self.base_pos_tracking.linearize(q, rd, vd)
            else:
                print("Planner of Type {} and Data type {} Not supported".format(planner.type, planner.ref_data_type))
                J = np.eye(self.robot.DoF)
                ed = np.zeros(self.robot.DoF)

            Js.append(J)
            eds.append(ed)
            task_types.append("Eq")

        qdot_d = self.hqp(Js, eds, task_types)

        return qdot_d, np.zeros(self.robot.DoF)

    def hqp(self, Js, eds, task_types):
        """ Cascaded QP for solving lexicographic quadratic programming problem

        :param Js:
        :param eds:
        :param task_types:
        :return:
        """
        Abar = cs.DM.zeros(0, self.QPsize)
        bbar = cs.DM.zeros(0)
        if task_types[0] == "Ineq":
            Cbar = Js[0]
            dbar = eds[0]
        else:
            Cbar = cs.DM.zeros(0, self.QPsize)
            dbar = cs.DM.zeros(0)

        for tid in range(len(task_types)):
            J = Js[tid]
            ed = eds[tid]

            opti = cs.Opti('conic')
            qdot = opti.variable(self.QPsize)
            w = opti.variable(J.shape[0])

            opti.minimize(0.5 * w.T @ w + 0.5 * self.params["ρ"] * qdot.T @ qdot)
            if Abar.shape[0] > 0:
                opti.subject_to(Abar @ qdot == bbar)
            if Cbar.shape[0] > 0:
                opti.subject_to(Cbar @ qdot <= dbar)
            if task_types[tid] == "Ineq":
                opti.subject_to(J @ qdot <= ed + w)
            else:
                opti.subject_to(J @ qdot == ed + w)

            p_opts = {"error_on_fail": True, "expand": True}
            s_opts = {"OutputFlag": 0, "LogToConsole": 0, "Presolve": 1, "BarConvTol": 1e-8,
                               "OptimalityTol": 1e-6}
            opti.solver('gurobi', p_opts, s_opts)

            sol = opti.solve()

            w_opt = sol.value(w)
            qdot_opt = sol.value(qdot)

            if task_types[tid] == "Ineq":
                Cbar = cs.vertcat(Cbar, J)
                dbar = cs.vertcat(dbar, ed + w_opt)
            else:
                Abar = cs.vertcat(Abar, J)
                bbar = cs.vertcat(bbar, ed + w_opt)

        return qdot_opt








