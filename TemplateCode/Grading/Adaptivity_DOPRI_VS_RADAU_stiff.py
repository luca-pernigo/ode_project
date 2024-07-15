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

# For latex text in matplotlib
plt.rcParams["text.usetex"] = True

# Load a problem from Problems.py
P = HIRES()

# Load one of the multiple Runge-Kutta methods from Steppers.py
solver_opts = {"nonlinear_solver": "scipy", "quasi_newton": False, "linear_solver": "direct", "matrix_free": False, "preconditioner": "none"}
RKs = []
try:
    RKs = RKs + [lambda adaptivity_opts: Radau(s=3, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts)]
except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
    print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
try:
    RKs = RKs + [lambda adaptivity_opts: DOPRI54(adaptivity_opts=adaptivity_opts)]
except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
    print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)


# Set initial value, maximal tolerance and how many tolerances to consider.
dt0 = 0.01
Tol_max = 1e-3
N_tols = 8

# No need to modify below this line ------------------------------------

# Define the list of time steps
Tols = np.array([Tol_max / 10**i for i in range(0, N_tols)])

yex = reference_solution(P, 1e-4, RKs)

data = {}
for RK in RKs:
    err = np.zeros(N_tols)
    cpu = np.zeros(N_tols)
    update_data = True
    for i in range(N_tols):
        adaptivity_opts = {"enable_adaptivity": True, "error_tolerance": Tols[i], "dt_safe_fac": 0.9, "dt_facmax": 5.0, "dt_facmin": 0.1}
        try:
            stepper = RK(adaptivity_opts)
            ti = TimeIntegrator(stepper, P, adaptivity_opts)
            t, y, et, n_steps, n_rejected_steps, avg_Newton_iter, avg_lin_solver_iter = ti.integrate(dt0, verbose=False)
            err[i] = np.linalg.norm(y[:, -1] - yex) / np.linalg.norm(1.0 + yex)
            cpu[i] = et
        except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
            print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
            update_data = False
            break
    if update_data:
        data[stepper.description] = [err, cpu]

fig, ax = plt.subplots()
# plot the error vs cpu time
fs_label = 20
fs_tick = 20
fs_legend = 20
fs_title = 24
markers = ["s", "v", "*", "x", "D", "d", "|", "_", ".", ",", "o", "p", "^", "<", ">", "1", "2", "3", "4", "h", "H", "+"]
for i, (stepper, (err, cpu)) in enumerate(data.items()):
    ax.loglog(err, cpu, label=stepper, marker=markers[i], markersize=8, linewidth=2.5)
ax.set_ylabel("CPU time [sec]", fontsize=fs_label, labelpad=-0.8)
ax.set_xlabel("$l^2$-norm error", fontsize=fs_label, labelpad=-0.8)
ax.tick_params(axis="x", labelsize=fs_tick)
ax.tick_params(axis="y", labelsize=fs_tick)
ax.legend(fontsize=fs_legend)
ax.set_title("Adaptivity, DOPRI_VS_RADAU: stiff problem", fontsize=fs_title)
plt.show()
fig.savefig("Grading/Adaptivity/stiff.eps", bbox_inches="tight", format="eps")
