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