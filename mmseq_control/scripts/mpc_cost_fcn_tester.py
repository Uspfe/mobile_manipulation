import rospy
import time
import numpy as np
import casadi as cs
import os
import matplotlib.pyplot as plt
from mmseq_control.MPCCostFunctions import RBF, RBFOld
import timeit


def rbf_plot(h, mu, zeta, axes):
    s = np.where(h < zeta, 1, 0)
    B = RBFOld.B_fcn(s, h, mu, zeta)
    dBdh = RBFOld.B_grad_fcn(s, h, mu, zeta)
    ddBddh = RBFOld.B_hess_fcn(s, h, mu, zeta)

    axes[0].plot(h, B, linewidth=2, label=f"$\mu = $ {mu}, $\zeta = $ {zeta}")
    axes[1].plot(h, dBdh, linewidth=2, label=f"$\mu = $ {mu}, $\zeta = $ {zeta}")
    axes[2].plot(h, ddBddh, linewidth=2, label=f"$\mu = $ {mu}, $\zeta = $ {zeta}")

    axes[0].plot(zeta, RBFOld.B_fcn(0, zeta, mu, zeta), "r.", markersize=8)
    axes[1].plot(zeta, RBFOld.B_grad_fcn(0, zeta, mu, zeta), "r.", markersize=8)
    axes[2].plot(zeta, RBFOld.B_hess_fcn(0, zeta, mu, zeta), "r.", markersize=8)


def rbfnew_plot(h, mu, zeta, axes):
    B = RBF.B_fcn(h, mu, zeta)
    dBdh = RBF.B_grad_fcn(h, mu, zeta)
    ddBddh = RBF.B_hess_fcn(h, mu, zeta)

    axes[0].plot(h, B, linewidth=2, label=f"new $\mu = $ {mu}, $\zeta = $ {zeta}")
    axes[1].plot(h, dBdh, linewidth=2, label=f"new $\mu = $ {mu}, $\zeta = $ {zeta}")
    axes[2].plot(h, ddBddh, linewidth=2, label=f"$new \mu = $ {mu}, $\zeta = $ {zeta}")

    axes[0].plot(zeta, RBF.B_fcn(zeta, mu, zeta), "r.", markersize=8)
    axes[1].plot(zeta, RBF.B_grad_fcn(zeta, mu, zeta), "r.", markersize=8)
    axes[2].plot(zeta, RBF.B_hess_fcn(zeta, mu, zeta), "r.", markersize=8)


def testRBFNew():
    mu = 0.001
    zeta = 0.005
    h = np.arange(start=-1, stop=1, step=0.01)
    fig_basic, axes_basic = plt.subplots(3, 1, sharex=True)
    rbf_plot(h, mu, zeta, axes_basic)
    rbfnew_plot(h, mu, zeta, axes_basic)

    axes_basic[0].set_title("B(h)")
    axes_basic[1].set_title("dBdh(h)")
    axes_basic[2].set_title("ddBddh(h)")
    plt.legend()
    plt.show(block=True)

    setup = """
from mmseq_control.MPCCostFunctions import RBF as RBFNew
from mmseq_control.MPCCostFunctions import RBFOld as RBF

import numpy as np
mu = 0.01
zeta = 0.01
h = np.arange(start=-1, stop=1, step=0.01)
"""
    print(
        "RBFNew Call {}".format(
            timeit.timeit("RBFNew.B_fcn(h, mu, zeta)", setup=setup, number=1000)
        )
    )

    print(
        "RBF Call {}".format(
            timeit.timeit(
                "RBF.B_fcn(np.where(h<zeta, 1, 0),h, mu, zeta)",
                setup=setup,
                number=1000,
            )
        )
    )


def testRBF():
    mu = 0.01
    zeta = 0.01
    h = np.arange(start=-1, stop=1, step=0.01)
    fig_basic, axes_basic = plt.subplots(3, 1, sharex=True)
    rbf_plot(h, mu, zeta, axes_basic)

    axes_basic[0].set_title("B(h)")
    axes_basic[1].set_title("dBdh(h)")
    axes_basic[2].set_title("ddBddh(h)")
    plt.legend()
    plt.show(block=False)

    mus = np.linspace(0.05, 0.25, num=int(0.5 / 0.05) + 1)
    zeta = 0.1
    fig_mu, axes_mu = plt.subplots(3, 1, sharex=True)
    for mu in mus:
        rbf_plot(h, mu, zeta, axes_mu)

    axes_mu[0].set_title("B(h)")
    axes_mu[1].set_title("dBdh(h)")
    axes_mu[2].set_title("ddBddh(h)")
    plt.legend()
    plt.show(block=False)

    zetas = np.linspace(0.05, 0.25, num=int(0.5 / 0.05) + 1)
    mu = 0.1
    fig_zeta, axes_zeta = plt.subplots(3, 1, sharex=True)
    for zeta in zetas:
        rbf_plot(h, mu, zeta, axes_zeta)

    axes_zeta[0].set_title("B(h)")
    axes_zeta[1].set_title("dBdh(h)")
    axes_zeta[2].set_title("ddBddh(h)")
    plt.legend()
    plt.show(block=True)


def testMPCCosts(config):
    from mmseq_control.robot import CasadiModelInterface
    from mmseq_control.MPCCostFunctions import (
        ControlEffortCostFunciton,
        EEPos3CostFunction,
        BasePos2CostFunction,
    )

    dt = 0.1
    N = 10
    # robot mdl
    casadi_model_interface = CasadiModelInterface(config["controller"])

    robot = casadi_model_interface.robot

    cost_base = BasePos2CostFunction(
        dt, N, robot, config["controller"]["cost_params"]["BasePos2"]
    )
    cost_eff = ControlEffortCostFunciton(
        dt, N, robot, config["controller"]["cost_params"]["Effort"]
    )
    cost_ee = EEPos3CostFunction(
        dt, N, robot, config["controller"]["cost_params"]["EEPos3"]
    )
    cost_fcn = cost_ee

    q = [0.0, 0.0, 0.0] + [0.0, -2.3562, -1.5708, -2.3562, -1.5708, 1.5708]
    v = np.zeros(9)
    x = np.hstack((np.array(q), v))
    QPsize = 18 * (N + 1) + 9 * N
    x_bar = np.tile(x, (N + 1, 1))
    u_bar = np.zeros((N, 9))
    r_bar = np.array([1] * cost_fcn.nr)
    r_bar = np.tile(r_bar, (N + 1, 1))
    # r_bar = np.ones(9)

    J_sym = cost_fcn.evaluate(x_bar, u_bar, r_bar)
    t0 = time.perf_counter()
    H_sym, g_sym = cost_fcn.quad(x_bar, u_bar, r_bar)
    t1 = time.perf_counter()
    print("Time{}".format(t1 - t0))
    g_num = np.zeros(cost_fcn.QPsize)
    eps = 1e-7

    for i in range(x_bar.shape[0]):
        for j in range(x_bar.shape[1]):
            x_bar_p = x_bar.copy()
            x_bar_p[i][j] += eps
            J_p = cost_fcn.evaluate(x_bar_p, u_bar, r_bar)
            g_num[i * x_bar.shape[1] + j] = (J_p - J_sym) / eps

    for i in range(u_bar.shape[0]):
        for j in range(u_bar.shape[1]):
            u_bar_p = u_bar.copy()
            u_bar_p[i][j] += eps
            J_p = cost_fcn.evaluate(x_bar, u_bar_p, r_bar)
            indx = (N + 1) * 18 + i * u_bar.shape[1] + j
            g_num[indx] = (J_p - J_sym) / eps

    print("Difference in gradient:{}".format(np.linalg.norm(g_num - g_sym)))
    print(H_sym)


def timeMPCCosts(config):
    from mmseq_control.robot import CasadiModelInterface
    from mmseq_control.MPCCostFunctions import (
        ControlEffortCostFunciton,
        ControlEffortCostFuncitonNew,
        EEPos3CostFunction,
        BasePos2CostFunction,
        SumOfCostFunctions,
    )

    dt = 0.1
    N = 10
    # robot mdl
    casadi_model_interface = CasadiModelInterface(config["controller"])

    robot = casadi_model_interface.robot

    cost_base = BasePos2CostFunction(
        dt, N, robot, config["controller"]["cost_params"]["BasePos2"]
    )
    cost_eff = ControlEffortCostFunciton(
        dt, N, robot, config["controller"]["cost_params"]["Effort"]
    )
    cost_ee = EEPos3CostFunction(
        dt, N, robot, config["controller"]["cost_params"]["EEPos3"]
    )
    cost_fcn = cost_ee

    cost_fcns = [cost_ee, cost_base, cost_eff]
    r_bars = []
    t_total = 0
    for cost_fcn in cost_fcns:
        x_bar = np.random.randn(N + 1, 18)
        u_bar = np.random.randn(N, 9)
        if cost_fcn.__class__.__name__ == "ControlEffortCostFuncitonNew":
            r_bar = np.random.randn(9)
        elif cost_fcn.__class__.__name__ == "ControlEffortCostFunciton":
            r_bar = np.random.randn(cost_fcns[-1].nr)
        else:
            r_bar = np.random.randn(N + 1, cost_fcn.nr)
        r_bars.append(r_bar)

        J_sym = cost_fcn.evaluate(x_bar, u_bar, r_bar)
        t0 = time.perf_counter()
        H_sym, g_sym = cost_fcn.quad(x_bar, u_bar, r_bar)
        t1 = time.perf_counter()
        t_total += t1 - t0
        print("Time {}: {}".format(cost_fcn.name, t1 - t0))
    print("Time Total: {}".format(t_total))

    t0 = time.perf_counter()
    sum_cost_functions = SumOfCostFunctions(cost_fcns)
    t1 = time.perf_counter()
    print("Construction Time Sum of Cost: {}".format(t1 - t0))

    t0 = time.perf_counter()
    H_sum, g_sum = sum_cost_functions.quad(x_bar, u_bar, *r_bars)
    t1 = time.perf_counter()
    print("Time Sum of Cost: {}".format(t1 - t0))


def timeQuad():
    setup = """
from mmseq_control.robot import  CasadiModelInterface
from mmseq_control.MPCCostFunctions import ControlEffortCostFunciton
from mmseq_utils import parsing
import numpy as np
config = parsing.load_config("/home/tracy/Projects/mm_slam/mm_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml")

dt = 0.1
N = 10
# robot mdl
casadi_model_interface = CasadiModelInterface(config["controller"])

robot = casadi_model_interface.robot

cost_eff = ControlEffortCostFunciton(dt, N, robot, config["controller"]["cost_params"]["Effort"])
cost_fcn = cost_eff

q = [0.0, 0.0, 0.0] + [0.0, -2.3562, -1.5708, -2.3562, -1.5708, 1.5708]
v = np.zeros(9)
x = np.hstack((np.array(q), v))
x_bar = np.tile(x, (N + 1, 1))
u_bar = np.zeros((N, 9))
r_bar = np.array([1] * cost_fcn.nr)
r_bar = np.tile(r_bar, (N + 1, 1))
    """

    print(
        "Normal call {}".format(
            timeit.timeit(
                "cost_fcn.quad(x_bar, u_bar, r_bar)", setup=setup, number=1000
            )
        )
    )


if __name__ == "__main__":
    from mmseq_utils import parsing

    config = parsing.load_config(
        "/home/tracy/Projects/mm_slam/mm_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml"
    )

    testRBFNew()
    # testRBFNew()
    # testMPCCosts(config)
    # timeMPCCosts(config)
    # timeQuad()
