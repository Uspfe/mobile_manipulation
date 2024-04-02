import rospy
import numpy as np
import casadi as cs
import os
import matplotlib.pyplot as plt
from mmseq_control.MPCCostFunctions import RBF

def testRBF():
    def rbf_plot(h, mu, zeta, axes):
        s = np.where(h<zeta, 1, 0)
        B = RBF.B_fcn(s, h, mu, zeta)
        dBdh = RBF.B_grad_fcn(s, h, mu, zeta)
        ddBddh = RBF.B_hess_fcn(s, h ,mu, zeta)

        axes[0].plot(h, B, linewidth=2, label=f"$\mu = $ {mu}, $\zeta = $ {zeta}")
        axes[1].plot(h, dBdh, linewidth=2, label=f"$\mu = $ {mu}, $\zeta = $ {zeta}")
        axes[2].plot(h, ddBddh, linewidth=2, label=f"$\mu = $ {mu}, $\zeta = $ {zeta}")

        axes[0].plot(zeta, RBF.B_fcn(0, zeta, mu, zeta), 'r.', markersize=8)
        axes[1].plot(zeta, RBF.B_grad_fcn(0, zeta, mu, zeta), 'r.', markersize=8)
        axes[2].plot(zeta, RBF.B_hess_fcn(0, zeta, mu, zeta), 'r.', markersize=8)

    mu = 0.01
    zeta = 0.01
    h = np.arange(start=-1, stop=1, step=0.01)
    fig_basic, axes_basic = plt.subplots(3,1,sharex=True)
    rbf_plot(h, mu, zeta, axes_basic)

    axes_basic[0].set_title("B(h)")
    axes_basic[1].set_title("dBdh(h)")
    axes_basic[2].set_title("ddBddh(h)")
    plt.legend()
    plt.show(block=False)

    mus = np.linspace(0.05, 0.25,num=int(0.5/0.05)+1)
    zeta = 0.1
    fig_mu, axes_mu = plt.subplots(3,1,sharex=True)
    for mu in mus:
        rbf_plot(h, mu, zeta, axes_mu)

    axes_mu[0].set_title("B(h)")
    axes_mu[1].set_title("dBdh(h)")
    axes_mu[2].set_title("ddBddh(h)")
    plt.legend()
    plt.show(block=False)

    zetas = np.linspace(0.05, 0.25,num=int(0.5/0.05)+1)
    mu = 0.1
    fig_zeta, axes_zeta = plt.subplots(3,1,sharex=True)
    for zeta in zetas:
        rbf_plot(h, mu, zeta, axes_zeta)

    axes_zeta[0].set_title("B(h)")
    axes_zeta[1].set_title("dBdh(h)")
    axes_zeta[2].set_title("ddBddh(h)")
    plt.legend()
    plt.show(block=True)   


if __name__ == "__main__":
    from mmseq_utils import parsing
    config = parsing.load_config("/home/tracy/Projects/mm_slam/mm_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml")
    testRBF()