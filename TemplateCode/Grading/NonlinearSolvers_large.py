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
P = Brusselator_1D(M=128)

adaptivity_opts = {"enable_adaptivity": False, "error_tolerance": 1e-4, "dt_safe_fac": 0.9, "dt_facmax": 5.0, "dt_facmin": 0.1}
solver_opts = [
    {"nonlinear_solver": "scipy", "quasi_newton": False},
    {"nonlinear_solver": "scipy", "quasi_newton": True},
    {"nonlinear_solver": "newton", "quasi_newton": False, "linear_solver": "direct", "matrix_free": False, "preconditioner": "none"},
    {"nonlinear_solver": "newton", "quasi_newton": True, "linear_solver": "direct", "matrix_free": False, "preconditioner": "none"},
    {"nonlinear_solver": "newton", "quasi_newton": False, "linear_solver": "iterative", "matrix_free": False, "preconditioner": "none"},
    {"nonlinear_solver": "newton", "quasi_newton": True, "linear_solver": "iterative", "matrix_free": False, "preconditioner": "none"},
    {"nonlinear_solver": "newton", "quasi_newton": False, "linear_solver": "iterative", "matrix_free": False, "preconditioner": "ILU"},
    {"nonlinear_solver": "newton", "quasi_newton": True, "linear_solver": "iterative", "matrix_free": False, "preconditioner": "ILU"},
    {"nonlinear_solver": "newton", "quasi_newton": False, "linear_solver": "iterative", "matrix_free": True, "preconditioner": "none"},
    {"nonlinear_solver": "newton", "quasi_newton": True, "linear_solver": "iterative", "matrix_free": True, "preconditioner": "none"},
]

# Define the steppers, but skip the ones that are not implemented
RK = Radau
RKs = []
for option in solver_opts:
    try:
        RKs = RKs + [RK(s=3, solver_opts=option, adaptivity_opts=adaptivity_opts)]
    except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
        print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)

if len(RKs) == 0:
    print(colorama.Fore.RED + "ERROR: No stepper has been implemented. Please implement at least one stepper." + colorama.Style.RESET_ALL)
    exit(1)

# Set how many time steps we want to consider in the efficiency experiment
# and the maximal one
dt_max = 0.1
N_dt = 5

# Define the list of time steps
dts = np.array([dt_max / 2**i for i in range(0, N_dt)])

yex = reference_solution(P, dts.min(), np.max([RK.order for RK in RKs]))

data = {}
for RK in RKs:
    print(f"Solving for {RK.description} ...")

    err = np.zeros(N_dt)
    cpu = np.zeros(N_dt)
    update_data = True
    for i in range(N_dt):
        try:
            ti = TimeIntegrator(RK, P, adaptivity_opts=adaptivity_opts)
            t, y, et, n_steps, n_rejected_steps, avg_Newton_iter, avg_lin_solver_iter = ti.integrate(dts[i], verbose=False)
            err[i] = np.linalg.norm(y[:, -1] - yex)
            cpu[i] = et
        except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
            print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
            update_data = False
            break
    if update_data:
        data[RK.description] = [err, cpu]

# Plot the results
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots(figsize=(6, 4))
# plot the error vs cpu time
markers = ["s", "v", "*", "x", "D", "d", "|", "_", ".", ",", "o", "p", "^", "<", ">", "1", "2", "3", "4", "h", "H", "+"]
for i, (stepper, (err, cpu)) in enumerate(data.items()):
    ax.loglog(err, cpu, label=stepper, marker=markers[i], markersize=8, linewidth=2.5)
# plot the error vs cpu time
fs_label = 20
fs_tick = 20
fs_legend = 20
fs_title = 24
ax.set_ylabel("CPU time [sec]", fontsize=fs_label, labelpad=-0.8)
ax.set_xlabel("$l^2$-norm error", fontsize=fs_label, labelpad=-0.8)
ax.tick_params(axis="x", labelsize=fs_tick)
ax.tick_params(axis="y", labelsize=fs_tick)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=fs_legend)
ax.set_title("Efficiency Nonlinear Solvers: large", fontsize=fs_title)
plt.show()
fig.savefig("Grading/NonlinearSolvers/large.eps", bbox_inches="tight", format="eps")
