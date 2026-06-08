import jax.random as jxr
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Optional, Sequence, Union, overload, override
from .solver import Solver

class ODESolver(Solver):

    # TODO:
    # use helper function for jit
    # or make this class a PyTree
    # see the jax and equinox doc


    def __init__(self, velocity_model: Callable,method='tsit5') -> None:
        super().__init__()
        self.velocity_model = velocity_model
        if method == "euler"
            self.solver = dfx.Euler()
        elif method == "midpoint":
            self.solver = dfx.Midpoint()
        elif method == "tsit5":
            self.solver = dfx.Tsit5()
        else:
            self.solver = None
            raise NotImplemented("Solver not available")

    @override
    def sample(
            self,
            key: PRNGKeyArray,
            x_init: Array,
            step_size: float,
            return_intermediates=False,
            **kwargs)->Union[Array, Sequence[Array]]:

        def ode_wrapper(t, x, args):
            return self.velocity_model(x=x, t=t, **kwargs)

        assert self.solver is not None

        term = dfx.ODETerm(ode_wrapper)
        x0 = jxr.normal(key,x_init.shape)
        sol = dfx.diffeqsolve(term,self.solver,t0=0.0,t1=1.0,dt0=step_size,y0=x0)

        if return_intermediates:
            return sol.ys
        else:
            return sol.ys[0]