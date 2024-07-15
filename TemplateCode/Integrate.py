import numpy as np
from Problems import YeastSugar, Kepler, HeatEquation, HIRES, ZombiesOutbreak, HarmonicOscillator, SIRV, Brusselator_2D, Brusselator_1D
from Steppers.PopularRK import ExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, Heun3, ThetaMethod1, ThetaMethod2
from Steppers.ExplicitRK import DOPRI54
from Steppers.ImplicitRK import LobattoIIIC, RandomCollocation, Gauss, Radau
from TimeIntegrator import TimeIntegrator
from ReferenceSolution import reference_solution
from Exceptions import IncompatibleOptions, UnknownOption
import colorama

# Load a problem from Problems.py
# P = YeastSugar()
# P = ZombiesOutbreak()
# P= SIRV
P=SIRV()

# Time step adaptivity options
adaptivity_opts = {
    "enable_adaptivity": True,  # Set to True for adaptive time stepping
    "error_tolerance": 1e-4,  # Error tolerance for adaptive time stepping
    "dt_safe_fac": 0.9,  # Safety factor for time step adjustment
    "dt_facmax": 5.0,  # Maximum factor by which time step can increase
    "dt_facmin": 0.1,  # Minimum factor by which time step can decrease
}
# Nonlinear solver options
solver_opts = {
    "nonlinear_solver": "scipy",  # Choose the nonlinear solver: scipy or newton
    "quasi_newton": False,  # Use quasi-Newton method if available
    "linear_solver": "direct",  # Choose the linear solver: direct or iterative
    "matrix_free": False,  # Use matrix-free methods (only for iterative linear solver)
    "preconditioner": "none",  # Choose the preconditioner: none or ILU. ILU is possible only in non matrix-free mode and with iterative linear solver
}

# We try to run the simulations, since many features are missing then its likely that an exception will be raised indicating what is wrong
try:
    # Load one of the Runge-Kutta methods
    # RK = Gauss(s=2, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts)
    RK = Radau(s=3, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts)
    # RK = DOPRI54(adaptivity_opts=adaptivity_opts)
    # RK = LobattoIIIC(solver_opts=solver_opts, adaptivity_opts=adaptivity_opts)
    # Choose a time step size
    dt = 0.1
    # Define the time integrator. It will solve the problem P using the chosen Runge-Kutta method (RK).
    ti = TimeIntegrator(RK, P, adaptivity_opts=adaptivity_opts)
    # Solve the problem using the defined time step (dt)
    t, y, et, n_steps, n_rejected_steps, avg_Newton_iter, avg_lin_solver_iter = ti.integrate(dt, verbose=False)
except (IncompatibleOptions, UnknownOption, NotImplementedError) as err:
    print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
    exit()

# Print integration results and statistics
print(f"Integrator: {RK.description}")
print(f"Solved in {et:0.8f} seconds")
print(f"Total steps: {n_steps} steps")
print(f"Accepted steps: {n_steps-n_rejected_steps} steps")
print(f"Rejected steps: {n_rejected_steps} steps")
print(f"Average Newton iterations per RK step: {avg_Newton_iter:.2f}")
print(f"Average linear solver iterations per Newton step: {avg_lin_solver_iter:.2f}")

# Print relative error if the exact solution is available
np.set_printoptions(precision=16)
yex = reference_solution(P, dt, RK.order, False)
if yex is not None:
    print(f"Relative error: {np.linalg.norm(yex-y[:,-1])/np.linalg.norm(yex)}")

# Display the computed values in a plot
P.show_solution(t, y)
