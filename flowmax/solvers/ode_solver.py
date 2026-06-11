import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jxr
import diffrax as dfx
import equinox as eqx
from einops import rearrange
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Sequence, Union
from jax.tree_util import register_pytree_node_class
from .solver import Solver

@register_pytree_node_class
class ODESolver:
    solver: dfx.AbstractSolver

    def __init__(self, backend):
        """
            A simple ODE solver class

            Args:
                backend: the diffrax ODE solver class we intent to use (Euler, Midpoint, Tsit5...)

        """
        super().__init__()
        self.solver = backend

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    # vmapable fn fot sample
    @eqx.filter_jit
    def _sample_fn(self, model, step_size, ts, rtol, atol, x_init)->Array:
        def vector_field(t, x, args):
            return model(x, t)
        term = dfx.ODETerm(vector_field)
        saveat = dfx.SaveAt(ts=ts)
        y0 = x_init
        if step_size is None:
            stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
            sol = dfx.diffeqsolve(term, self.solver,
                t0=0,
                t1=1,
                dt0=None,
                stepsize_controller=stepsize_controller,
                y0=y0,
                saveat=saveat
            )
        else:
            sol = dfx.diffeqsolve(term, self.solver,
                t0=0,
                t1=1,
                dt0=step_size,
                y0=y0,
                saveat=saveat)
        return sol.ys

    def sample(
            self,
            velocity_model,
            x_init: Array,
            step_size: float|None,
            timegrid: Array = jnp.array([0.0,1.0]),
            return_intermediates: bool = False,
            rtol: float =1e-5,
            atol: float =1e-5,
            )->Union[Array, Sequence[Array]]:

        sample_fn = ft.partial(self._sample_fn, velocity_model, step_size, timegrid, rtol, atol)
        samples = jax.vmap(sample_fn)(x_init)
        # rearange to put the time dim in first place
        samples = rearrange(samples, "b t c -> t b c")

        if return_intermediates:
            return samples
        else:
            return samples[-1]




    # ------------------------------------------------------------------ #
    # Pytree utils
    # ------------------------------------------------------------------ #
    def tree_flatten(self):
        children = ()
        aux_data = {"backend":self.solver}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)