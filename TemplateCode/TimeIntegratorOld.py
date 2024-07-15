import time
from tqdm import tqdm
import numpy as np
import colorama


# This class implements a time integrator for the ODE y' = f(t,y)
class TimeIntegrator:
    def __init__(self, stepper, problem, adaptivity_opts):
        self.S = stepper  # the Runge-Kutta method
        self.P = problem  # the problem to be solved
        self.enable_adaptivity = adaptivity_opts["enable_adaptivity"]  # enable/disable adaptivity
        self.err_tol = adaptivity_opts["error_tolerance"]  # the error tolerance
        self.dt_safe_fac = adaptivity_opts["dt_safe_fac"]  # the safety factor for the time step
        self.dt_facmin = adaptivity_opts["dt_facmin"]  # the minimum time step factor
        self.dt_facmax = adaptivity_opts["dt_facmax"]  # the maximum time step factor

    def integrate(self, dt, verbose=False):
        """
        Args:
            dt (float): step size
            verbose (bool, optional): Enable/disable verbose output. Defaults to False. If False, a progress bar is shown.

        Returns:
            t (np.ndarray): the time points
            y (np.ndarray): the solution at time points t
            et (float): elapsed time
            n_steps (int): number of accepted steps
            n_rejected_steps (int): number of rejected steps
            avg_Newton_iter (float): average number of Newton iterations per step
            avg_lin_solver_iter (float): average number of linear solver iterations per Newton iteration
        """

        alloc_size = int(1e3)  # initial allocation size
        n_vars = self.P.y0.shape[0]  # number of variables

        # Allocate space for y and t
        y = np.zeros((n_vars, alloc_size), order="F")
        t = np.zeros(alloc_size)

        # Initialize the solution with the initial value
        y[:, 0] = self.P.y0
        t0 = self.P.t0
        t[0] = t0
        tend = self.P.Tend

        # some nicknames for convenience
        step = self.S.step
        f = self.P.f

        # initialize the progress bar
        if not verbose:
            bar_total = 1000
            pbar = tqdm(desc="Time", total=bar_total)  # a nice bar showing progress

        last = False  # flag indicating whether the last step was performed
        n_it = 0  # number of accepted steps, also current index in y and t
        n_rejected_steps = 0  # number of rejected steps
        n_steps = 0  # iteration counter for the time loop
        tot_Newton_iter = 0  # total number of Newton iterations
        tot_lin_solver_iter = 0  # total number of linear solver iterations

        tic = time.perf_counter()  # start measuring time

        # main time loop
        while not last:
            # check whether the next step would exceed the final time and adjust the step size accordingly
            if t[n_it] + dt > tend:
                dt = tend - t[n_it]
                last = True

            # check whether we need to allocate more space
            if n_it + 1 == y.shape[1]:
                y.resize(n_vars, n_it + alloc_size, refcheck=False)
                t.resize(n_it + alloc_size, refcheck=False)


            if self.enable_adaptivity==False:
                # call the time stepper, which returns an approximative solution at time t[n_it]+dt
                # compute E^_n+1
                y[:, n_it + 1], err, n_Newton_iter, n_lin_solver_iter = step(t[n_it], y[:, n_it], f, dt)
                t[n_it + 1] = t[n_it] + dt
                # update the statistics
                tot_Newton_iter += n_Newton_iter
                tot_lin_solver_iter += n_lin_solver_iter
                n_steps += 1

                if verbose:
                    print(f"t = {t[n_it+1]:.3e}, dt = {dt:.3e}, ||y||={np.linalg.norm(y[:,n_it+1]):.3e}, err = {err:.3e}", flush=True)
                else:
                    pbar.n = np.round((t[n_it] - t0) / (tend - t0) * bar_total)
                    
                    pbar.refresh()

    
                n_it += 1

            # if adaptivity tru
            elif self.enable_adaptivity==True:
                self.get_dt_new(self, err, n_steps)
                


                

        toc = time.perf_counter()  # end measuring time
        et = toc - tic

        # resize the arrays to the actual size
        y.resize(n_vars, n_it + 1, refcheck=False)
        t.resize(n_it + 1, refcheck=False)

        if not verbose:
            pbar.close()  # close the bar

        # compute the average number of Newton and linear solver iterations
        avg_Newton_iter = tot_Newton_iter / n_steps
        if tot_Newton_iter > 0:
            avg_lin_solver_iter = tot_lin_solver_iter / tot_Newton_iter
        else:
            avg_lin_solver_iter = 0

        return t, y, et, n_steps, n_rejected_steps, avg_Newton_iter, avg_lin_solver_iter

    def get_dt_new(self, dt, err, n_steps):
        """Adapt the time step

        Args:
            dt (float): the current time step
            err (float): the error estimate
            n_steps (int): the number of steps performed so far

        Returns:
            dt_new (float): the new time step
        """
        # raise NotImplementedError("TimeIntegrator.get_dt_new() is not implemented.")


        factol=1.1
        
        y[:, n_it + 1], err, n_Newton_iter, n_lin_solver_iter = step(t[n_it], y[:, n_it], f, dt)
        p_hat=self.S.phat
        # ok
        if n_it==0:
            print("first iteration")
            dt_new=self.dt_safe_fac*(self.err_tol/err)**(1/(p_hat+1))*dt
            print("dt_new candidate at first iteration", dt_new)
            # time.sleep(5)
        
        else:
            if n_it==1:
                print("second iteration")
                print("t",t[:n_it+1])
                # time.sleep(5)

                # iteration 1 t=[t0, t1]
                # h0
                h_n_minus_1=t[n_it]-t[n_it-1]
                # default value we cannot go backwards more
                h_n_minus_2=0.1
                
                # pass h0 get E^_1
                _, err_n_minus_1, _,_ =step(t[n_it-1], y[:, n_it-1], f, h_n_minus_1)
                err_n_minus_2=0
            else:
                
                h_n_minus_1=t[n_it]-t[n_it-1]
                h_n_minus_2=t[n_it-1]-t[n_it-2]

                # compute E^_n
                _, err_n_minus_1, _,_ =step(t[n_it-1], y[:, n_it-1], f, h_n_minus_1)
                # compute E^_n-1
                _, err_n_minus_2, _,_=step(t[n_it-2], y[:, n_it-2], f, h_n_minus_2)
                
            
            dt_new=(err_n_minus_2/err_n_minus_1)**(1/(p_hat+1))*(h_n_minus_1/h_n_minus_2)*(self.err_tol/err_n_minus_1)**(1/(p_hat+1))*h_n_minus_1


        dt_new=min(max(dt_new, self.dt_facmin*dt), self.dt_facmax*dt)
        
        
        print("move",err,factol*self.err_tol)
        # time.sleep(5)
        if err<=factol*self.err_tol:
            print("We move next")
            # time.sleep(1)

            # we make the step
            t[n_it + 1] = t[n_it] + dt
            # hn+1
            dt=dt_new
            n_it+=1
            if verbose:
                print(f"t = {t[n_it+1]:.3e}, dt = {dt:.3e}, ||y||={np.linalg.norm(y[:,n_it+1]):.3e}, err = {err:.3e}", flush=True)
        
            else:
                pbar.n = np.round((t[n_it] - t0) / (tend - t0) * bar_total)
                pbar.refresh()
            

        else:
            print("Reject step")
            print("err E^_n+1", err, "at step", n_it)
            print(err)
            
            
            # hn
            print("dt", dt, "dt_new", dt_new)
            # time.sleep(5)
            dt=dt_new
            n_rejected_steps+=1
        
        # update the statistics
        # this every time we do a step
        tot_Newton_iter += n_Newton_iter
        tot_lin_solver_iter += n_lin_solver_iter

        
        # n_steps are total steps
        # n_rejected are the rejected
        # n_it are the accepted steps
        n_steps += 1
        print("Steps",n_steps)
        # time.sleep(1)        




        return dt_new
