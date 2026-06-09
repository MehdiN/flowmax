from types import NotImplementedType
import jax.random as jxr
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Optional, Sequence, Union, overload, override
from jax.tree_util import register_pytree_node_class
from .solver import Solver


@register_pytree_node_class
class ODESolver:

    def __init__(self, velocity_model: Callable,method='tsit5') -> None:
        super().__init__()
        self.velocity_model = velocity_model
        if method == "euler":
            self.solver = dfx.Euler()
        elif method == "midpoint":
            self.solver = dfx.Midpoint()
        elif method == "tsit5":
            self.solver = dfx.Tsit5()
        else:
            self.solver = None
            raise NotImplementedType("Solver not available")

    @eqx.filter_jit
    def sample(
            self,
            key: PRNGKeyArray,
            x_init: Array,
            step_size: float,
            timegrid = jnp.array([0.0,1.0]),
            return_intermediates=False,
            **kwargs)->Union[Array, Sequence[Array]]:

        def ode_wrapper(t, x, args):
            return self.velocity_model(x=x, t=t, **kwargs)

        assert self.solver is not None
        # check if t0 and t1 are included in the timegrid ?
        saveat = dfx.SaveAt(t0=True,t1=True,ts=timegrid)
        term = dfx.ODETerm(ode_wrapper)
        x0 = jxr.normal(key, x_init.shape)
        sol = dfx.diffeqsolve(term,self.solver,t0=0.0,t1=1.0,dt0=step_size,y0=x0,saveat=saveat)

        if return_intermediates:
            return sol.ys
        else:
            return sol.ys[0]

    def tree_flatten(self):
        children = ()
        aux_data = {"velocity_model": self.velocity_model,"solver":self.solver}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)