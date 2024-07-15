import numpy as np
from Steppers.ExplicitRK import DOPRI54
from Steppers.ImplicitRK import Gauss, Radau
from TimeIntegrator import TimeIntegrator
import pathlib


def reference_solution(P, dt_min, p, compute=True):
    """
    Obtain a reference solution for the given problem.

    Parameters:
    - P: Problem class representing the ordinary differential equation.
    - dt_min: Minimum time step size (see compute_reference_solution)
    - p: Order of accuracy (see compute_reference_solution)
    - compute: If True then compute a reference solution if not found. Else return None.

    Returns:
    - Reference solution for the problem.
    """
    # Here we look for a reference solution to problem P
    # First in the problem class P and then on a file on disk.
    # If not found then we compute one
    if hasattr(P, "yex"):
        print("For this problem we have an exact solution :) !")
        return P.yex
    elif pathlib.Path("ReferenceSolutions/" + P.__class__.__name__).with_suffix(".bin").exists():
        return np.fromfile(pathlib.Path("ReferenceSolutions/" + P.__class__.__name__).with_suffix(".bin"))
    elif compute:
        return compute_reference_solution(P, dt_min, p)
    else:
        return None


def compute_reference_solution(P, dt_min, p):
    """
    Compute a reference solution to be compared against solutions
    computed with order p methods (or less) and step size dt_min
    (or higher).
    Here we will make some choices to ensure that the referene
    solution is more accurate than the other solutions.
    Still, for some problems it is useful to come back and tweak the options.

    Parameters:
    - P: Problem class representing the ordinary differential equation.
    - dt_min: Minimum time step size.
    - p: Order of accuracy.

    Returns:
    - Reference solution for the problem.
    """

    # Determine the appropriate stepper based on the problem classification
    if P.classification == "stiff":
        s = int(np.ceil((p + 1) / 2))
        adaptivity_opts = {"enable_adaptivity": False, "error_tolerance": 1e-4, "dt_safe_fac": 0.9, "dt_facmax": 5.0, "dt_facmin": 0.1}
        solver_opts = {"nonlinear_solver": "newton", "quasi_newton": False, "linear_solver": "direct", "matrix_free": False, "preconditioner": "none"}
        stepper = Radau(s, solver_opts, adaptivity_opts)
        dt = dt_min / 8.0
    elif P.classification == "geometric":
        s = int(np.ceil(p / 2))
        adaptivity_opts = {"enable_adaptivity": False, "error_tolerance": 1e-4, "dt_safe_fac": 0.9, "dt_facmax": 5.0, "dt_facmin": 0.1}
        solver_opts = {"nonlinear_solver": "newton", "quasi_newton": False, "linear_solver": "direct", "matrix_free": False, "preconditioner": "none"}
        stepper = Gauss(s, solver_opts, adaptivity_opts)
        dt = dt_min / 8.0
    elif P.classification == "easy":
        adaptivity_opts = {"enable_adaptivity": False, "error_tolerance": 1e-4, "dt_safe_fac": 0.9, "dt_facmax": 5.0, "dt_facmin": 0.1}
        stepper = DOPRI54(adaptivity_opts)
        dt = dt_min ** (p / stepper.order) / 8.0
    else:
        raise Exception("Unknown problem classification.")

    print(f"Computing a reference solution with {stepper.description} and dt = {dt}")
    ti = TimeIntegrator(stepper, P, adaptivity_opts)
    t, y_ref, et, n_steps, n_rejected_steps, avg_Newton_iter, avg_lin_solver_iter = ti.integrate(dt=dt, verbose=False)

    y_ref = y_ref[:, -1]

    # Write the reference solution to file for future use
    y_ref.tofile(pathlib.Path("ReferenceSolutions/" + P.__class__.__name__).with_suffix(".bin"))

    return y_ref
