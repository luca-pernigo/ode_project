import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../TemplateCode")

from Problems import YeastSugar, Kepler, HeatEquation, HIRES, ZombiesOutbreak, HarmonicOscillator, SIRV, Brusselator_1D, Brusselator_2D
from Steppers.PopularRK import ExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, Heun3
from Steppers.ExplicitRK import DOPRI54
from Steppers.ImplicitRK import LobattoIIIC, RandomCollocation, Gauss, Radau
from TimeIntegrator import TimeIntegrator
from ReferenceSolution import reference_solution
from Exceptions import IncompatibleOptions, UnknownOption
import colorama


# Load a problem from Problems.py
P = HIRES()

# Load one of the multiple Runge-Kutta methods from Steppers.py
solver_opts = {"nonlinear_solver": "scipy", "quasi_newton": False, "linear_solver": "direct", "matrix_free": False, "preconditioner": "none"}
RK = lambda adaptivity_opts: Radau(s=3, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts)


# Set initial value, maximal tolerance and how many tolerances to consider.
dt0 = 0.01
Tol_max = 1e-1
N_tols = 10

# Set maximal dt and how many dts to consider
dt_max = 1.0
N_dts = 4

# No need to modify below this line ------------------------------------

# Define the list of time steps and tolerances
Tols = np.array([Tol_max / 10**i for i in range(0, N_tols)])
dts = np.array([dt_max / 2**i for i in range(0, N_dts)])

yex = reference_solution(P, dts.min(), 5)

err_enable = np.zeros(N_tols)
cpu_enable = np.zeros(N_tols)
for i in range(N_tols):
    adaptivity_opts = {"enable_adaptivity": True, "error_tolerance": Tols[i], "dt_safe_fac": 0.9, "dt_facmax": 5.0, "dt_facmin": 0.1}
    try:
        stepper = RK(adaptivity_opts)
        ti = TimeIntegrator(stepper, P, adaptivity_opts)
        t, y, et, n_steps, n_rejected_steps, avg_Newton_iter, avg_lin_solver_iter = ti.integrate(dt0, verbose=False)
        err_enable[i] = np.linalg.norm(y[:, -1] - yex) / np.linalg.norm(1.0 + yex)
        cpu_enable[i] = et
    except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
        print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
        break

err_disable = np.zeros(N_dts)
cpu_disable = np.zeros(N_dts)
adaptivity_opts = {"enable_adaptivity": False, "error_tolerance": 1.0, "dt_safe_fac": 0.9, "dt_facmax": 5.0, "dt_facmin": 0.1}
for i in range(N_dts):
    try:
        stepper = RK(adaptivity_opts)
        ti = TimeIntegrator(stepper, P, adaptivity_opts)
        t, y, et, n_steps, n_rejected_steps, avg_Newton_iter, avg_lin_solver_iter = ti.integrate(dts[i], verbose=False)
        err_disable[i] = np.linalg.norm(y[:, -1] - yex) / np.linalg.norm(1.0 + yex)
        cpu_disable[i] = et
    except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
        print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
        break

# For latex text in matplotlib
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()
# plot the error vs cpu time
fs_label = 20
fs_tick = 20
fs_legend = 20
fs_title = 24

ax.loglog(err_enable, cpu_enable, label="Enabled", marker="o", markersize=8, linewidth=2.5)
ax.loglog(err_disable, cpu_disable, label="Disabled", marker="x", markersize=8, linewidth=2.5)
ax.set_ylabel("CPU time [sec]", fontsize=fs_label, labelpad=-0.8)
ax.set_xlabel("$l^2$-norm error", fontsize=fs_label, labelpad=-0.8)
ax.tick_params(axis="x", labelsize=fs_tick)
ax.tick_params(axis="y", labelsize=fs_tick)
ax.legend(fontsize=fs_legend)
ax.set_title("Adaptivity: Enable VS Disable", fontsize=fs_title)
plt.show()
fig.savefig("Grading/Adaptivity/enable_vs_disable.eps", bbox_inches="tight", format="eps")