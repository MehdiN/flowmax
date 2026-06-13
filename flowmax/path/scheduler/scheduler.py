from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union
from jaxtyping import Array
import jax
import jax.numpy as jnp

@jax.tree_util.register_dataclass
@dataclass
class SchedulerOutput:

    alpha_t: Array
    sigma_t: Array
    d_alpha_t: Array
    d_sigma_t: Array

@jax.tree_util.register_pytree_node_class
class Scheduler(ABC):
    """
        Base class
    """

    @abstractmethod
    def __call__(self, t: Array) -> SchedulerOutput:
        """
        Args:
            t (Array): time in [0,1]
        Returns:
            SchedulerOutput
        """

    @abstractmethod
    def snr_inverse(self, snr: Array)->Array:
        """
            TODO
        """

    def tree_flatten(self):
        children = None
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)    


class ConvexScheduler(Scheduler):
    @abstractmethod
    def __call__(self, t: Array) -> SchedulerOutput:
        """
            TODO
        """

    @abstractmethod
    def kappa_inverse(self, kappa: Array)->Array:
        """
            TODO
        """

    def snr_inverse(self, snr: Array)->Array:
        kappa_t = snr / (1.0 + snr)
        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """
        Conditional Optimal Transport Scheduler
    """

    def __call__(self, t: Array) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1-t,
            d_alpha_t=jnp.ones_like(t),
            d_sigma_t=-jnp.ones_like(t)
        )

    def kappa_inverse(self, kappa: Array) -> Array:
        return kappa

class PolynomialConvexScheduler(ConvexScheduler):
    n: float | int
    
    """
        Polynomial Scheduler

        Args:
        n: Scalar-like, polynomial power must be > 0
    
    """

    def __init__(self, n: float | int) -> None:

        self.n = n

    def __call__(self, t: Array) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t= t ** self.n,
            sigma_t= 1 - t ** self.n,
            d_alpha_t= self.n * (t ** (self.n - 1)),
            d_sigma_t= - self.n * (t ** (self.n - 1)),
        )

    def kappa_inverse(self, kappa: Array) -> Array:
        return jnp.pow(kappa, 1.0 / self.n)


class CosineScheduler(Scheduler):
    """
        Cosine Scheduler
    """
    def __call__(self, t: Array) -> SchedulerOutput:
        pi = jnp.pi
        return SchedulerOutput(
            alpha_t= jnp.sin(pi / 2 * t),
            sigma_t=jnp.cos(pi /2 * t),
            d_alpha_t= pi / 2 * jnp.cos(pi /2 * t),
            d_sigma_t= -pi / 2 * jnp.sin(pi / 2 * t)
        )

    def snr_inverse(self, snr: Array) -> Array:
        return 2.0 * jnp.atan(snr) / jnp.pi


class VPScheduler(Scheduler):
    bate_min: float
    beta_max: float

    """
        Variance Preserving Scheduler
        
        Args:
         beta_min: float
         beta_max: float
    """

    def __init__(self, beta_min: float = 0.1, beta_max:float = 20.)->None:
        self.beta_min = beta_min
        self.beta_max = beta_max


    def __call__(self, t: Array) -> SchedulerOutput:
        b = self.beta_min
        B = self.beta_max
        T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
        dT = - (1-t) * (B - b) - b

        return SchedulerOutput(
            alpha_t=jnp.exp(- 0.5 * T),
            sigma_t = jnp.sqrt( 1 - jnp.exp(-0.5 * T)),
            d_alpha_t= - 0.5 * dT * jnp.exp(-0.5 * T),
            d_sigma_t= 0.5 * dT * jnp.exp(-T) / jnp.sqrt(1 - jnp.exp(-T))
        )

    def snr_inverse(self, snr: Array) -> Array:
        T = - jnp.log(snr ** 2 / (snr ** 2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = 1 - ((-b + jnp.sqrt(b**2 + 2 * (B - b) * T)) / (B - b))
        return t

class LinearVPScheduler(Scheduler):
    
    """
        Linear Variance Preserving Scheduler
    """

    def __call__(self, t: Array) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t = t,
            sigma_t = jnp.sqrt(1 - t**2),
            d_alpha_t = jnp.ones_like(t),
            d_sigma_t = - t / jnp.sqrt(1 - t ** 2)
        )

    def snr_inverse(self, snr: Array) -> Array:
        return jnp.sqrt(snr**2 / (1 + snr**2))
