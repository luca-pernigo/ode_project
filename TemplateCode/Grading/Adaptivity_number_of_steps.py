import numpy as np
import sys
sys.path.append('../TemplateCode')

from Problems import YeastSugar, Kepler, HeatEquation, HIRES, ZombiesOutbreak, HarmonicOscillator, SIRV, Brusselator_2D, Brusselator_1D
from Steppers.PopularRK import ExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, Heun3, ThetaMethod1, ThetaMethod2
from Steppers.ExplicitRK import DOPRI54
from Steppers.ImplicitRK import LobattoIIIC, RandomCollocation, Gauss, Radau
from TimeIntegrator import TimeIntegrator
from ReferenceSolution import reference_solution
from Exceptions import IncompatibleOptions, UnknownOption
import colorama

# Load a problem from Problems.py
P = HIRES()

# Time step adaptivity options
adaptivity_opts = {"enable_adaptivity": True, "error_tolerance": 1e-9, "dt_safe_fac": 0.9, "dt_facmax": 5.0, "dt_facmin": 0.1}
# Nonlinear solver options
solver_opts = {"nonlinear_solver": "scipy", "quasi_newton": False, "linear_solver": "direct", "matrix_free": False, "preconditioner": "none"}

try:
    # Load one of the Runge-Kutta methods
    RK = Radau(s=3, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts)
    # Choose a time step size
    dt = 1e-3
    # Define the time integrator. It will solve P using RK.
    ti = TimeIntegrator(RK, P, adaptivity_opts=adaptivity_opts)
    # and solve using time step dt
    t, y, et, n_steps, n_rejected_steps, avg_Newton_iter, avg_lin_solver_iter = ti.integrate(dt, verbose=False)
except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
    print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
    exit()

print(f"Integrator: {RK.description}")
print(f"Solved in {et:0.8f} seconds")
print(f"Total steps: {n_steps} steps")
print(f"Accepted steps: {n_steps-n_rejected_steps} steps")
print(f"Rejected steps: {n_rejected_steps} steps")
print(f"Average Newton iterations per RK step: {avg_Newton_iter:.2f}")
print(f"Average linear solver iterations per Newton step: {avg_lin_solver_iter:.2f}")

np.set_printoptions(precision=16)
yex = reference_solution(P, dt, RK.order, False)
if yex is not None:
    print(f"Relative error: {np.linalg.norm(yex-y[:,-1])/np.linalg.norm(yex)}")

# show the computed values in a plot
P.show_solution(t, y)
