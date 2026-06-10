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

    def sample(
            self,
            velocity_model,
            x_init: Array,
            step_size: float|None,
            timegrid: Array = jnp.array([0.0,1.0]),
            return_intermediates: bool =False,
            rtol: float =1e-5,
            atol: float =1e-5,
            )->Union[Array, Sequence[Array]]:

        sample_fn = ft.partial(self._sample_vmap_fn, velocity_model, step_size, timegrid, rtol, atol)
        samples = jax.vmap(sample_fn)(x_init)
        # rearange to put the time dim in first place
        samples = rearrange(samples, "b t c -> t b c")

        if return_intermediates:
            return samples
        else:
            return samples[-1]

    @eqx.filter_jit
    def _sample_vmap_fn(self, model, step_size, ts, rtol, atol, x_init)->Array:
        def ode_wrapper(t, x, args):
            return model(x, t)
        term = dfx.ODETerm(ode_wrapper)
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

    # def compute_likelihood(
    #     self,
    #     velocity_model,
    #     x_1: Array,
    #     log_p0: Callable,
    #     step_size: float | None,
    #     timegrid = jnp.array([1.0,0.0]),
    #     return_intermediates = False,
    #     exact_divergence: bool = False,
    #     enable_grad: bool = False,
    #     *,
    #     key
    #     ):

    #     assert timegrid[0] == 1.0 and timegrid[-1] == 0.0

    #     # Fix the random projection for the Hutchinson divergence estimator
    #     if not exact_divergence:
    #         # z = (jxr.normal(key, x_1.shape) < 0) * 2.0 - 1
    #         z = jxr.rademacher(key, x_1.shape).astype(x_1.dtype)
    #     else:
    #         z = jnp.zeros(x_1.shape)

    #     def dynamics_fn(t, y, z):
    #         xt = y[:-1]
    #         ut = velocity_model(xt, t)

    #         fun = lambda x: velocity_model(x, t)
    #         if exact_divergence:
    #             # compute the exact divergence
    #             jacobian = jax.jacobian(fun)(xt)
    #             div = jnp.trace(jacobian,axis1=-2,axis2=-1)
    #         else:
    #             _, ret = jax.jvp(fun, (xt, ), (z, ))
    #             div = jnp.dot(ret, z)

    #         if not enable_grad:
    #             ut = jax.lax.stop_gradient(ut)
    #             div = jax.lax.stop_gradient(div)
    #         div = jnp.atleast_1d(div)
    #         return jnp.concatenate([ut, -div])

    #     compute_likelihood_fn = ft.partial(self._compute_likelihood_fn, dynamics_fn, step_size, timegrid)
    #     _y = jnp.zeros((x_1.shape[0], 1),dtype=jnp.float32)
    #     y0 = jnp.concatenate([x_1, _y], axis=-1)
    #     ys = jax.vmap(compute_likelihood_fn)(y0,z)
    #     ys = rearrange(ys, "b t c -> t b c")
    #     final_ys = ys[-1]
    #     x_source = final_ys[...,:-1]
    #     log_det = final_ys[...,-1]
    #     source_log_p = log_p0(x_source)

    #     total_log_p = source_log_p + log_det

    #     if return_intermediates:
    #         return ys[...,:-1], total_log_p
    #     else:
    #         return x_source, total_log_p


    # @eqx.filter_jit
    # def _compute_likelihood_fn(self, dynamic_fn, step_size, ts, y0, z)->Array:
    #     def ode_wrapper(t, x, args):
    #         z = args
    #         return dynamic_fn(t, x, z)
    #     term = dfx.ODETerm(ode_wrapper)
    #     saveat = dfx.SaveAt(ts=ts)
    #     args = z
    #     dt = - jnp.fabs(step_size)
    #     sol = dfx.diffeqsolve(
    #         terms=term,
    #         solver=self.solver,
    #         t0=1.0,
    #         t1=0.0,
    #         dt0=dt,
    #         y0=y0,
    #         saveat=saveat,
    #         args=args
    #         )
    #     return sol.ys


    def tree_flatten(self):
        children = ()
        aux_data = {"backend":self.solver}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)