from scheduler.scheduler import Scheduler, CondOTScheduler
from src.path import BaseProbPath
from src.path import PathSample
from jaxtyping import Array


class AffineProbPath(BaseProbPath):
    
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def sample(self, x_0: Array, x_1: Array, t: Array, *, key=None)->PathSample:
        """
            TODO
        """
        scheduler_output = self.scheduler(t)
        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t
        d_alpha_t = scheduler_output.d_alpha_t
        d_sigma_t = scheduler_output.d_sigma_t

        # construct xt ~ p_t(x|x1)
        x_t = sigma_t * x_0 + alpha_t * x_1
        dx_t = d_sigma_t * x_0 + d_alpha_t * x_1

        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)

    def target_to_velocity(self, x_1: Array, x_t: Array, t: Array)->Array:
        """
            TODO
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = d_sigma_t / sigma_t
        b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t

        return a_t * x_t + b_t * x_1

    def epsilon_to_velocity(self, epsilon: Array, x_t: Array, t: Array)->Array:


        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t

        a_t = 1 / alpha_t
        b_t = -sigma_t / alpha_t

        return a_t * x_t + b_t * epsilon

    def velocity_to_epsilon(self, velocity:Array, x_t: Array, t:Array)->Array:

        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = -d_alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)
        b_t = alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)

        return a_t * x_t + b_t * velocity


    def target_to_epsilon(self, x_1:Array, x_t:Array, t: Array)->Array:
        
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t

        a_t = 1 / sigma_t
        b_t = -alpha_t / sigma_t

        return a_t * x_t + b_t * x_1


class CondOTProbPath(AffineProbPath):
    """
        Conditional Optimal Transport probability path
    """
    def __init__(self):
        self.scheduler = CondOTScheduler()