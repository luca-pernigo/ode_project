import numpy as np
import scipy.optimize

# This file implements one step of some popular Runge-Kutta methods
# For the description of the return values, see ExplicitRK.py


# The Explicit Euler method
class ExplicitEuler:
    order = 1
    description = "ExplicitEuler"

    def step(self, t, y, f, dt):
        ynew = y + dt * f(t, y)
        return ynew, 0, 0, 0


# The Implicit Euler method
class ImplicitEuler:
    order = 1
    description = "ImplicitEuler"

    def step(self, t, y, f, dt):
        def g(x):
            return x - y - dt * f(t + dt, x)

        ynew, info, ier, msg = scipy.optimize.fsolve(g, y, full_output=True)
        return ynew, 0, info["nfev"], 0


# The Implicit Midpoint method
class ImplicitMidpoint:
    order = 2
    description = "ImplicitMidpoint"

    def step(self, t, y, f, dt):
        def g(x):
            return x - y - dt * f(t + dt / 2.0, (x + y) / 2.0)

        ynew, info, ier, msg = scipy.optimize.fsolve(g, y, full_output=True)
        return ynew, 0, info["nfev"], 0


# The Explicit Midpoint method
class ExplicitMidpoint:
    order = 2
    description = "ExplicitMidpoint"

    def step(self, t, y, f, dt):
        ynew = y + dt * f(t + 0.5 * dt, y + 0.5 * dt * f(t, y))
        return ynew, 0, 0, 0


class Heun3:
    order = 3
    description = "Heun3"

    def step(self, t, y, f, dt):
        k1 = f(t, y)
        k2 = f(t + dt / 3.0, y + dt * k1 / 3.0)
        k3 = f(t + 2.0 * dt / 3.0, y + 2.0 * dt * k2 / 3.0)
        ynew = y + dt * (k1 + 3.0 * k3) / 4.0
        return ynew, 0, 0, 0


# The theta-method (variant 1)
class ThetaMethod1:
    description = "ThetaMethod1"

    def __init__(self, theta=0.5):
        self.theta = theta
        self.order = 1 if theta != 0.5 else 2

    def step(self, t, y, f, dt):
        def g(x):
            return x - y - dt * (self.theta * f(t + dt, x) + (1.0 - self.theta) * f(t, y))

        ynew, info, ier, msg = scipy.optimize.fsolve(g, y, full_output=True)
        return ynew, 0, info["nfev"], 0


# The theta-method (variant 2)
class ThetaMethod2:
    description = "ThetaMethod2"

    def __init__(self, theta=0.5):
        self.theta = theta
        self.order = 1 if theta != 0.5 else 2

    def step(self, t, y, f, dt):
        def g(x):
            return x - y - dt * f(t + self.theta * dt, y + self.theta * (x - y))

        ynew, info, ier, msg = scipy.optimize.fsolve(g, y, full_output=True)
        return ynew, 0, info["nfev"], 0
