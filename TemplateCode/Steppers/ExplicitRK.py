import numpy as np
from Exceptions import IncompatibleOptions, UnknownOption
import time

# Here we implement a general class for explicit Runge-Kutta methods
# Then a child class for the DOPRI54 method


class ExplicitRK:
    """
    This class implements a general explicit Runge-Kutta method
    """

    def __init__(self, A, b, c, adaptivity_opts):
        """
        Initialize the Runge-Kutta method

        Args:
            A (np.array or list of lists): Runge-Kutta matrix in (s,s) shape array or list of lists
            b (np.array or list): Runge-Kutta weights b in shape (s,) array or list
            c (np.array or list): Runge-Kutta nodes c in shape (s,) array or list
            adaptivity_opts (dict): dictionary containing the options for time step adaptivity
        """
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = np.array(c)
        self.s = self.b.shape[0]  # number of stages

        # Check that the Runge-Kutta matrix is lower triangular
        if not np.allclose(self.A, np.tril(self.A)):
            raise IncompatibleOptions("The Runge-Kutta matrix is not lower triangular")

        self.adaptivity_opts = adaptivity_opts

        # Set the description of the method, used for printing and plotting
        self.description = self.__class__.__name__

    def step(self, t0, y0, f, dt):
        """
        Performs one step of the Runge-Kutta method

        Args:
            t0 (float): the initial time
            y0 (np.array): the initial value
            f (function): the right-hand side of the ODE
            dt (float): the time step size

        Returns:
            y1 (np.array): the solution at time t0+dt
            err (float): the estimated error
            0 (int): number of Newton iterations (not used for explicit methods)
            0 (int): number of linear solver iterations (not used for explicit methods)
        """

        # Allocate space for the stages, done only once
        # we have s stages each of dimension dx1, the dimension of y
        if not hasattr(self, "k"):
            self.k = [np.zeros_like(y0) for i in range(self.s)]

        # raise NotImplementedError("ExplicitRK.step() is not implemented")

        # Compute the stages k and y1
        # f is loaded from the problem instance P.f
        # f takes as argument t and y
    

        # update/fill with a for loop the vector of ki
        for i in range(self.s):
            sum_aij=0

            # for explicit methods we do not have non linearity
            # so this is enough
            # this for loop computes the sum in ki involving all terms kj with j<i
            for j in range(i):
                sum_aij+=self.A[i,j]*self.k[j]
            
            self.k[i]=f(t0+self.c[i]*dt, y0+dt*sum_aij)


        # create with a for loop the vector dx1 y1
        y1=np.zeros_like(y0)
        for i in range(self.s):
            y1+=dt*self.b[i]*self.k[i]
        y1+=y0

        # Estimate the error
        if self.adaptivity_opts["enable_adaptivity"]:
            err = self.estimate_error(self.k, t0, y0, f, dt)
        else:
            err = self.adaptivity_opts["error_tolerance"]

        return y1, err, 0, 0

    def estimate_error(self, k, t0, y0, f, dt):
        """
        Estimate the error of the RK method at time t0+dt
        This method is overridden in child classes whenever they do have an error estimator

        Args:
            z (np.array): Array of shape (s*n_vars) containing the solution of the RK system Phi(z)=0
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size

        Raises:
            NotImplementedError: When the error estimator is not implemented for the method

        """
        
        raise NotImplementedError("Error estimator not implemented for " + self.description)


class DOPRI54(ExplicitRK):
    def __init__(self, adaptivity_opts):
        # Define the Butcher tableau
        A = [
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ]
        c = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
        # b order 5
        b = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
        
        # order 4 b hat
        # b = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]

        # Set the order of the method
        self.order = 5

        super().__init__(A, b, c, adaptivity_opts)

        # Define the error estimator coefficients
        self.bhat = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]

    def estimate_error(self, k, t0, y0, f, dt):
        """
        Estimate the error of the solution at t0+dt

        Args:
            k (list of np.array): the stages k_i
            t0 (float): current time
            y0 (np.array): initial value
            f (function): the right-hand side of the ODE
            dt (float): the time step size

        Returns:
            err (float): the estimated error
        """
        
        # Compute the error estimate E_n+1

        # order of embedded method
        self.phat = 4
        # same dimension as y-> Rd
        diff=np.zeros_like(y0)

        # Compute diff = ^y_n+1 - y_n+1
        for i in range(len(k)):
            diff+=dt*k[i]*(self.bhat[i]-self.b[i])

        # raise NotImplementedError("DOPRI54.estimate_error() is not implemented")

        

        # Estimate the error. Divide by sqrt(diff.size) to make it independent of the number of variables
        err = np.linalg.norm(diff) 

        # I normalize it in the TimeIntegrator class
        # /np.sqrt(self.P.y0.size)

        return err
