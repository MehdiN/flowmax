import functools as ft
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
from einops import rearrange
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Optional, Sequence, Union
from jax.tree_util import register_pytree_node_class
from .solver import Solver



@register_pytree_node_class
class ODESolver:
    solver: dfx.AbstractSolver

    def __init__(self, solver):
        """
            TODO: usage doc
            maybe pass the dfx solver as an argument and 
            use TsiT5 as default if None
        """
        super().__init__()
        self.solver = solver

    # @eqx.filter_jit
    def sample(
            self,
            velocity_model,
            x_init: Array,
            step_size: float,
            timegrid = jnp.array([0.0,1.0]),
            return_intermediates=False
            )->Union[Array, Sequence[Array]]:

        # assert self.solver is not None
        # saveat = dfx.SaveAt(t0=False,t1=False,ts=timegrid)
        # term = dfx.ODETerm(ode_wrapper)
        # x0 = jxr.normal(key, x_init.shape)
        # sol = dfx.diffeqsolve(term,self.solver,t0=0.0,t1=1.0,dt0=step_size,y0=x0,saveat=saveat)

        sample_fn = ft.partial(self._sample_vmap_fn, velocity_model, step_size, timegrid)
        samples = jax.vmap(sample_fn)(x_init)
        # rearange to put the time dim in first place
        samples = rearrange(samples, "b t c -> t b c")

        if return_intermediates:
            return samples
        else:
            return samples[-1]

    @eqx.filter_jit
    def _sample_vmap_fn(self, model, step_size, ts, x_init)->Array:
        def ode_wrapper(t, x, args):
            return model(x, t)
        term = dfx.ODETerm(ode_wrapper)
        saveat = dfx.SaveAt(t0=False,t1=False,ts=ts)
        x_0 = x_init
        sol = dfx.diffeqsolve(term, self.solver, t0=0,t1=1,dt0=step_size,y0=x_0,saveat=saveat)
        return sol.ys


    def tree_flatten(self):
        children = ()
        aux_data = {"solver":self.solver}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)