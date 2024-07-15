import numpy as np


import matplotlib.pyplot as plt

# For latex text in matplotlib
plt.rcParams["text.usetex"] = True


# This file contains a list of ordinary differential equations
class YeastSugar:
    def __init__(self):
        # Define the initial value, the integration interval and come coefficients
        self.y0 = np.array([4.0, 0.1])
        self.t0 = 0
        self.Tend = 10.0  # in days
        self.alpha = 16.0  # billions of yeas cells produced by one billion of yeast cells per kg of sugar per liter in one day
        self.beta = 1.0  # billions of dying yeast cells per billion of yeast cells per day if sugar is over
        self.gamma1 = 5.0
        self.gamma2 = 500.0
        self.delta = 0.05  # kg of sugar per liter eaten by one billion yeast cells per day

        # an 'exact' solution
        # If you change Tend then this solution is not valid any more
        self.yex = np.array([0.0976821825333738, 0.0053450960856697])

        # classify the problem (easy, stiff, geometric)
        self.classification = "easy"

    def f(self, t, y):
        # The right hand side of y'=f(t,y)
        fy = np.zeros_like(y)
        fy[0] = self.alpha * y[0] * y[1] * 0.5 * (1.0 - np.tanh(self.gamma1 * (y[0] - 10.0))) - self.beta * y[0] * 0.5 * (1.0 - np.tanh(self.gamma2 * (y[1] - 0.01)))
        fy[1] = -self.delta * y[0] * y[1]
        return fy

    def show_solution(self, t, y):
        # Shows the computed solution
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(t, y[0, :], label="yeast")
        ax[1].plot(t, y[1, :], label="sugar")
        ax[0].legend()
        ax[1].legend()
        plt.show()


class SIRV:
    def __init__(self):
        # Define the initial value, the integration interval and come coefficients
        N = 1000
        self.y0 = np.array([N - 1, 1, 0, 0])
        self.t0 = 0
        self.Tend = 20.0
        self.beta = 3.18e-3
        self.gamma = 0.44
        self.lmbda = 0.1

        self.classification = "easy"

    def f(self, t, y):
        # The right hand side of y'=f(t,y)
        fy = np.zeros_like(y)
        fy[0] = -self.beta * y[0] * y[1] - self.lmbda * y[0]
        fy[1] = self.beta * y[0] * y[1] - self.gamma * y[1]
        fy[2] = self.gamma * y[1]
        fy[3] = self.lmbda * y[0]
        return fy

    def show_solution(self, t, y):
        # Shows the computed solution
        fig, ax = plt.subplots(5, 1)
        ax[0].plot(t, y[0, :], label="S")
        ax[1].plot(t, y[1, :], label="I")
        ax[2].plot(t, y[2, :], label="R")
        ax[3].plot(t, y[3, :], label="V")
        ax[4].plot(t, np.sum(y, axis=0), label="S+I+R+V")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[3].legend()
        ax[4].legend()
        plt.show()


# Example 1.6 from "Griffiths, D. F.; Higham., D. J. (2010). Numerical Methods for Ordinary Differentail Equations. Springer-Verlag."
class ZombiesOutbreak:
    def __init__(self):
        # Define the initial value, the integration interval and come coefficients
        self.y0 = np.array([500.0, 10.0, 0.0])
        self.t0 = 0
        self.Tend = 4.0
        self.alpha = 0.005
        self.beta = 0.01
        self.zeta = 0.02

        self.classification = "easy"

    def f(self, t, y):
        # The right hand side of y'=f(t,y)
        fy = np.zeros_like(y)
        fy[0] = -self.beta * y[0] * y[1]
        fy[1] = self.beta * y[0] * y[1] + self.zeta * y[2] - self.alpha * y[0] * y[1]
        fy[2] = -self.zeta * y[2] + self.alpha * y[0] * y[1]
        return fy

    def show_solution(self, t, y):
        # Shows the computed solution
        fig, ax = plt.subplots()
        ax.plot(t, y[0, :], label="humans")
        ax.plot(t, y[1, :], label="zombies")
        ax.plot(t, y[2, :], label="removed zombies")
        ax.legend()
        plt.show()


class HeatEquation:
    def __init__(self, M, add_noise=False):
        self.x0 = 0.0
        self.x1 = 5.0
        self.M = M
        self.dx = (self.x1 - self.x0) / self.M
        self.x = np.linspace(self.x0, self.x1, M + 1)
        self.y0 = np.exp(self.x / 5.0 - 1.0) * np.sin(3.0 * np.pi * self.x / 10.0) ** 2
        if add_noise:
            dB = [0]
            np.random.seed(1989)
            dB += [np.random.normal(loc=0.0, scale=self.dx) for i in range(M)]
            B = 1 * np.cumsum(dB)
            noise = (self.x1 - self.x) * B
            self.y0 += noise
        self.t0 = 0.0
        self.Tend = 1.0

        self.classification = "stiff"

    def f(self, t, y):
        fy = -2.0 * y / self.dx / self.dx
        fy[0] += 2.0 * y[1] / self.dx / self.dx
        fy[1:-1] += y[2:] / self.dx / self.dx
        fy[-1] += 2.0 * y[-2] / self.dx / self.dx
        fy[1:-1] += y[0:-2] / self.dx / self.dx
        fy += y - y**2
        return fy

    def show_solution(self, t, y):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        T, X = np.meshgrid(t, self.x)
        ax.plot_surface(T, X, y, cmap=plt.get_cmap("viridis"), linewidth=0, antialiased=False)
        plt.show()


class Kepler:
    def __init__(self, periods=2):
        self.e = 0.6
        self.y0 = np.array([1.0 - self.e, 0.0, 0.0, np.sqrt((1.0 + self.e) / (1.0 - self.e))])
        self.t0 = 0.0
        self.p = periods  # periods
        self.Tend = self.p * 2.0 * np.pi
        self.yex = self.y0

        self.classification = "geometric"

    def f(self, t, y):
        fy = np.zeros_like(y)
        fy[0:2] = y[2:4]
        fy[2:4] = -y[0:2] / np.linalg.norm(y[0:2]) ** 3
        return fy

    def show_solution(self, t, y):
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(
            y[0, :],
            y[1, :],
            label="trajectory",
        )
        ax[0].legend()

        def H(q, p):
            return 0.5 * np.linalg.norm(p, axis=0) ** 2 - 1.0 / np.linalg.norm(q, axis=0)

        q = y[0:2, :]
        p = y[2:4, :]
        ax[1].plot(t, H(q, p), label="Energy")
        ax[1].legend()
        plt.show()


class HarmonicOscillator:
    def __init__(self, periods=2):
        self.x0 = 1.0
        self.v0 = 1.0
        self.m = 1.0
        self.k = 1.0
        self.omega = np.sqrt(self.k / self.m)
        self.y0 = np.array([self.x0, self.v0])
        self.t0 = 0.0
        self.p = periods
        self.Tend = self.p * 2.0 * np.pi / self.omega
        self.yex = self.y0

        self.classification = "geometric"

    def f(self, t, y):
        fy = np.zeros_like(y)
        fy[0] = y[1]
        fy[1] = -self.omega**2 * y[0]
        return fy

    def show_solution(self, t, y):
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(
            t,
            y[0, :],
            label="position",
        )
        ax[0].plot(
            t,
            y[1, :],
            label="velocity",
        )
        ax[0].legend()

        def H(y1, y2):
            return 0.5 * y2**2 + 0.5 * self.omega**2 * y1**2

        ax[1].plot(t, H(y[0, :], y[1, :]), label="Energy")
        ax[1].legend()
        plt.show()


# The famous stiff benchmark problem
class HIRES:
    def __init__(self):
        self.t0 = 0.0
        self.Tend = 321.8122
        self.y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057])
        self.yex = np.array(
            [0.000737131257332567, 0.000144248572631618, 0.000058887297409676, 0.001175651343283149, 0.002386356198831330, 0.006238968252742796, 0.002849998395185769, 0.002850001604814231]
        )

        self.classification = "stiff"

    def f(self, t, y):
        fy = np.zeros_like(y)
        fy[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007
        fy[1] = 1.71 * y[0] - 8.75 * y[1]
        fy[2] = -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4]
        fy[3] = 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3]
        fy[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6]
        fy[5] = -280 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6]
        fy[6] = 280 * y[5] * y[7] - 1.81 * y[6]
        fy[7] = -280 * y[5] * y[7] + 1.81 * y[6]
        return fy

    def show_solution(self, t, y):
        fig, ax = plt.subplots(8, 1)
        for i in range(8):
            ax[i].plot(
                t,
                y[i, :],
                label=f"$y_{i}$",
            )
            ax[i].legend()
        plt.show()


# another very famous benchmark problem
class Brusselator_1D:
    def __init__(self, M):
        self.M = M
        self.dx = 1.0 / self.M
        self.x = np.linspace(0.0, 1.0, M + 1)
        self.x = self.x[:-1]
        self.y0 = np.concatenate((np.sin(4.0 * np.pi * self.x) ** 2, np.sin(2.0 * np.pi * self.x)))

        self.alpha = 1e-2

        self.t0 = 0.0
        self.Tend = 1.0

        self.classification = "stiff"

    def f(self, t, y):
        u, v = y[0 : self.M], y[self.M : 2 * self.M]
        fy = self.alpha * np.concatenate((self.laplacian(u), self.laplacian(v)))
        if t >= 1.1:
            force = 5 * ((self.x - 0.3) ** 2 < 0.01)
        else:
            force = 0.0
        fy[0 : self.M] += 1.0 + u**2 * v - 4.4 * u + force
        fy[self.M : 2 * self.M] += 3.4 * u - u**2 * v

        return fy

    def laplacian(self, y):
        n = self.M
        v = np.zeros(n + 2)
        v[1:-1] = y
        v[0] = y[-1]
        v[-1] = y[0]
        Ly = (v[:-2] + v[2:] - 2 * y) / self.dx**2
        return Ly

    def show_solution(self, t, y):
        n_vars = 2
        fig, axs = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"})

        T, X = np.meshgrid(t, self.x)
        for var in range(n_vars):
            axs[var].plot_surface(T, X, y[var * self.M : (var + 1) * self.M, :], cmap=plt.get_cmap("viridis"), linewidth=0, antialiased=False)
            axs[var].set_xlabel("x")
            axs[var].set_ylabel("t")
            axs[var].set_title(f"var {var}")

        plt.show()


class Brusselator_2D:
    def __init__(self, M):
        self.M = M
        self.Msq = M**2
        self.dx = 1.0 / self.M
        self.x = np.linspace(0.0, 1.0, M + 1)
        self.x = self.x[:-1]
        self.X, self.Y = np.meshgrid(self.x, self.x)
        self.y0 = np.concatenate((22.0 * np.array(self.Y * (1.0 - self.Y) ** (3.0 / 2.0)).ravel(), 27.0 * np.array(self.X * (1 - self.X) ** (3.0 / 2.0)).ravel()))

        self.alpha = 1e-2

        self.t0 = 0.0
        self.Tend = 3.0

        self.classification = "stiff"

    def f(self, t, y):
        u, v = y[0 : self.Msq], y[self.Msq : 2 * self.Msq]
        fy = self.alpha * np.concatenate((self.laplacian(u), self.laplacian(v)))
        if t >= 1.1:
            force = 5 * np.array(((self.X - 0.3) ** 2 + (self.Y - 0.6) ** 2 < 0.01)).ravel()
        else:
            force = 0.0
        fy[0 : self.Msq] += 1.0 + u**2 * v - 4.4 * u + force
        fy[self.Msq : 2 * self.Msq] += 3.4 * u - u**2 * v

        return fy

    def laplacian(self, y):
        n = self.M
        v = np.zeros((n + 2, n + 2))
        y = y.reshape((n, n))
        v[1:-1, 1:-1] = y
        v[0, 1:-1] = y[-1, :]
        v[-1, 1:-1] = y[0, :]
        v[1:-1, 0] = y[:, -1]
        v[1:-1, -1] = y[:, 0]
        Ly = (v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:] - 4 * y) / self.dx**2
        y = y.ravel()
        return Ly.ravel()

    def show_solution(self, t, y):
        n_out = 4
        n_vars = 2
        fig, axs = plt.subplots(nrows=n_vars, ncols=n_out, subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(self.x, self.x)

        ind_out = np.round(np.linspace(0, y.shape[1] - 1, n_out)).astype(int)
        for var in range(n_vars):
            for i in range(n_out):
                axs[var][i].plot_surface(X, Y, y[var * self.Msq : (var + 1) * self.Msq, ind_out[i]].reshape((self.M, self.M)), cmap=plt.get_cmap("viridis"), linewidth=0, antialiased=False)
                axs[var][i].set_xlabel("x")
                axs[var][i].set_ylabel("y")
                axs[var][i].set_title(f"var {var}, t={t[ind_out[i]]}")

        plt.show()
