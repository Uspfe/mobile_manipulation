#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:34:21 2021

@author: tracy
"""

import numpy as np
from scipy.linalg import block_diag
from qpsolvers import solve_qp

epsilon = 1e-5
from cvxopt.solvers import qp
from cvxopt import solvers
from cvxopt import matrix, sparse
import osqp
from numpy import inf
from scipy import sparse
from mosek import iparam
import time


class PrioritizedLinearSystems():

    @staticmethod
    def getSubsetEq(A, b, Abar=None, bbar=None, Cbar=None, dbar=None, rho=epsilon, init_vals=None):
        n = A.shape[1]
        P = np.matmul(A.transpose(), A) + np.eye(n) * rho
        q = -np.matmul(b.transpose(), A)
        # x = solve_qp(P, q, Cbar, dbar, Abar, bbar, solver='osqp', verbose=False, eps_abs=1e-2, initvals=init_vals)
        # x = solve_qp(P, q, Cbar, dbar, Abar, bbar, solver='mosek', verbose=False)

        ## Use osqp
        # prob = osqp.OSQP()
        # Ap = np.vstack((Abar, Cbar))
        # l = np.hstack((bbar, -inf*np.ones(dbar.shape)))
        # u = np.hstack((bbar, dbar))
        # prob.setup(sparse.csc_matrix(P), q, sparse.csc_matrix(Ap), l, u, eps_abs=1e-4, eps_rel=1e-5, max_iter=20000, verbose=True)
        # results = prob.solve()
        # x = results.x
        # print(x)
        # if x[0] is None:
        #     print(1)

        ## Using cvxopt
        P = matrix(P)
        q = matrix(q)
        if Cbar is not None:
            G = matrix(Cbar)
        else:
            G = None
        if dbar is not None:
            h = matrix(dbar)
        else:
            h = None
        if Abar is not None:
            Ap = matrix(Abar)
        else:
            Ap = None
        if bbar is not None:
            bp = matrix(bbar)
        else:
            bp = None
        solvers.options['show_progress'] = False
        # t0 = time.perf_counter()
        if init_vals is not None:
            initval = {'x': matrix(init_vals)}
        else:
            initval = None

        results = qp(P, q, G, h, Ap, bp, initvals=initval)
        # t1 = time.perf_counter()
        # print("Prioritized: {}".format(t1-t0))
        x = np.array(results['x']).squeeze()
        if Abar is None:
            return (A, np.matmul(A, x), Cbar, dbar), x
        else:
            Anew = np.vstack((Abar, A))
            bnew = np.hstack((bbar, np.matmul(A, x)))

            return (Anew, bnew, Cbar, dbar), x

    @staticmethod
    def getSubsetIneq(C, d, Abar=None, bbar=None, Cbar=None, dbar=None, rho=epsilon):
        n = C.shape[1]
        m = C.shape[0]

        P = block_diag(rho * np.eye(n), np.eye(m))
        q = np.zeros(m + n)
        G1 = np.hstack((C, -np.eye(m)))
        if Cbar is None:
            G2 = np.zeros((0, m + n))
        else:
            G2 = np.hstack((Cbar, np.zeros((Cbar.shape[0], m))))
        G3 = np.hstack((np.zeros((m, n)), -np.eye(m)))
        G = np.vstack((G1, G2, G3))
        if dbar is None:
            h = np.hstack((d, np.zeros((m))))
        else:
            h = np.hstack((d, dbar, np.zeros((m))))

        if Abar is None:
            A = Abar
            b = bbar
        else:
            A = np.hstack((Abar, np.zeros((Abar.shape[0], m))))
            b = bbar
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        if A is None:
            Ap = None
            bp = None
        else:
            Ap = matrix(A)
            bp = matrix(b)

        results = qp(P, q, G, h, Ap, bp)
        z = np.array(results['x']).squeeze()
        x = z[:n]
        # z = solve_qp(P, q, G, h, A, b, verbose=True)
        # x = z[:n]
        w = np.matmul(C, x) - d
        eq_index = np.where(w > 3e-4)[0]
        ineq_index = np.where(w <= 3e-4)[0]

        if Abar is None:
            Anew = C[eq_index]
            bnew = np.matmul(C[eq_index], x)
        else:
            Anew = np.vstack((Abar, C[eq_index]))
            bnew = np.hstack((bbar, np.matmul(C[eq_index], x)))

        if Cbar is None:
            Cnew = C[ineq_index]
            dnew = d[ineq_index]
        else:
            Cnew = np.vstack((Cbar, C[ineq_index]))
            dnew = np.hstack((dbar, d[ineq_index]))
        return (Anew, bnew, Cnew, dnew), x


class PrioritizedLinearSystemsNew:

    @staticmethod
    def getSubsetEq(A, b, Abar=None, bbar=None, Cbar=None, dbar=None, rho=epsilon, init_vals=None):
        n = A.shape[1]
        P = np.matmul(A.transpose(), A) + np.eye(n) * rho
        q = -np.matmul(b.transpose(), A)
        # eps_abs = 1e-4
        # x = solve_qp(P, q, Cbar, dbar, Abar, bbar, solver='osqp', verbose=False, eps_abs=eps_abs, initvals=init_vals)
        # if x is None:
        #     for i in range(1):
        #         eps_abs*=10
        #         x = solve_qp(P, q, Cbar, dbar, Abar, bbar, solver='osqp', verbose=False, eps_abs=eps_abs, initvals=init_vals)
        #         if x is not None:
        #             break
        # x = solve_qp(P, q, Cbar, dbar, Abar, bbar, solver='mosek', verbose=False)

        ## Use osqp
        # prob = osqp.OSQP()
        # Ap = np.vstack((Abar, Cbar))
        # l = np.hstack((bbar, -inf*np.ones(dbar.shape)))
        # u = np.hstack((bbar, dbar))
        # prob.setup(sparse.csc_matrix(P), q, sparse.csc_matrix(Ap), l, u, eps_abs=1e-4, eps_rel=1e-5, max_iter=20000, verbose=True)
        # results = prob.solve()
        # x = results.x
        # print(x)
        # if x[0] is None:
        #     print(1)

        ## Using cvxopt
        P = matrix(P)
        q = matrix(q)
        if len(Cbar) == 0:
            G = None
            h = None
        else:
            G = matrix(Cbar)
            h = matrix(dbar)

        if len(Abar) == 0:
            Ap = None
            bp = None
        else:
            Ap = matrix(Abar)
            bp = matrix(bbar)

        solvers.options['show_progress'] = False
        # t0 = time.perf_counter()
        initval = {'x': matrix(init_vals)}
        results = qp(P, q, G, h, Ap, bp, initvals=initval)
        # t1 = time.perf_counter()
        # print("Prioritized: {}".format(t1-t0))
        x = np.array(results['x']).squeeze()

        w = np.abs(np.matmul(A, x) - b)
        Cnew = np.vstack((Cbar, A, -A))
        dnew = np.hstack((dbar, w + b, w - b))

        return (Abar, bbar, Cnew, dnew), x

#
# if __name__ == "__main__":
#     from PrioritizedFramework import PrioritizedLinearSystems as PLS
#
#     Abar = np.array([[1., 1.], [1., -1.]])
#     bbar = np.array([1., -1.])
#     A = np.array([[1., 0.]])
#     b = np.array([1.])
#     linsys, _ = PLS.getSubsetEq(A, b, np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2)), np.zeros(0))
#
#     C = np.array([[1, 0]])
#     d = np.array([-1])
#     Abar = None
#     bbar = None
#     Cbar = np.array([[-1, 0]])
#     dbar = np.array([-2])
#     (Abar, bbar, Cbar, dbar), _ = PLS.getSubsetIneq(C, d, Abar, bbar, Cbar, dbar)
#
#     C = np.array([[1, 0]])
#     d = np.array([-2])
#     (Abar, bbar, Cbar, dbar), _ = PLS.getSubsetIneq(C, d, Abar, bbar, Cbar, dbar)