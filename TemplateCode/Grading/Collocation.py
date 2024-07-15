import numpy as np
import matplotlib.pyplot as plt
import sys
# add the directory TemplateCode where all python files needed are located
# python3 Grading/Collocation.py
sys.path.append('../TemplateCode')

from Problems import YeastSugar, Kepler, HeatEquation, HIRES, ZombiesOutbreak, HarmonicOscillator, SIRV, Brusselator_1D, Brusselator_2D
from Steppers.PopularRK import ExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, Heun3
from Steppers.ExplicitRK import DOPRI54
from Steppers.ImplicitRK import LobattoIIIC, RandomCollocation, Gauss, Radau
from TimeIntegrator import TimeIntegrator
from ReferenceSolution import reference_solution
from Exceptions import IncompatibleOptions, UnknownOption
import colorama


# Load a problem from Problems.py
P = SIRV()

# Load one of the multiple Runge-Kutta methods from Steppers.py
adaptivity_opts = {"enable_adaptivity": False, "error_tolerance": 1e-4, "dt_safe_fac": 0.9, "dt_facmax": 5.0, "dt_facmin": 0.1}
solver_opts = {"nonlinear_solver": "scipy", "quasi_newton": False, "linear_solver": "direct", "matrix_free": False, "preconditioner": "none"}
RKs = []

# Try to initialize RandomCollocation, Radau and Gauss methods. If they are not implemented then print an error message and skip them
try:
    RKs = RKs + [RandomCollocation(s=s, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts) for s in range(1, 4)]
except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
    print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
try:
    RKs = RKs + [Radau(s=s, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts) for s in range(1, 4)]
except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
    print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
try:
    RKs = RKs + [Gauss(s=s, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts) for s in range(1, 4)]
except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
    print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)

# Set how many time steps we want to consider in the efficiency experiment
# and the maximal one
dt_max = 0.05
N_dt = 5

# No need to modify below this line ------------------------------------

# Define the list of time steps
dts = np.array([dt_max / 2**i for i in range(0, N_dt)])

yex = reference_solution(P, dts.min(), RKs)

for RK in RKs:
    # Define the time integrator. It will solve P using RK.
    ti = TimeIntegrator(RK, P, adaptivity_opts)
    print(f"Solving for {RK.description} ...")

    err = np.zeros(N_dt)
    cpu = np.zeros(N_dt)
    ended_with_error = False
    for i in range(N_dt):
        try:
            t, y, et, n_steps, n_rejected_steps, avg_Newton_iter, avg_lin_solver_iter = ti.integrate(dts[i], verbose=False)
            err[i] = np.linalg.norm(y[:, -1] - yex) / np.linalg.norm(1.0 + yex)
            cpu[i] = et
        except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
            print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
            ended_with_error = True
            break

    if ended_with_error:
        continue

    plt.rcParams["text.usetex"] = True
    fig, ax = plt.subplots()
    # plot the error
    ax.loglog(dts, err, label="Err", marker="o", markersize=10, linewidth=3)
    p = RK.order
    ax.loglog(dts, err[0] / dts[0] ** p * dts**p, linestyle="dashed", color="black", linewidth=3, label=f"$O(\Delta t^{p})$")
    fs_label = 22
    fs_tick = 22
    fs_legend = 22
    fs_title = 26
    ax.set_xlabel("$h$", fontsize=fs_label, labelpad=-0.8)
    ax.set_ylabel("$l^2$-norm error", fontsize=fs_label, labelpad=-0.8)
    ax.tick_params(axis="x", labelsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.legend(fontsize=fs_legend)
    ax.set_title(RK.description, fontsize=fs_title)
    # plt.show()
    fig.savefig("Grading/Collocation/" + RK.description.replace(" ", "_") + ".eps", bbox_inches="tight", format="eps")
