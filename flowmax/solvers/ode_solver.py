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
    # Likelihood
    # ------------------------------------------------------------------ #

    @staticmethod
    def _div_jacobian(u_fn: Callable, xi: Array)->Array:
        J = jax.jacobian(u_fn)(xi)
        d = xi.size
        return jnp.trace(J.reshape(d,d))

    @staticmethod
    def _div_hutchinson(u_fn: Callable, xi: Array, zi: Array) -> Array:
        _,Jz = jax.jvp(u_fn, (xi,), (zi,))
        return jnp.dot(zi.flatten(), Jz.flatten())


    @eqx.filter_jit
    def _compute_likelihood_fn(self, dynamic_fn, exact_divergence, step_size, atol, rtol, saveat, x_1, key)->Array:
        if exact_divergence:
            z = jnp.zeros_like(x_1)
        else:
            z = jxr.rademacher(key, x_1.shape).astype(jnp.float32)
        args = z
        # define the vector field
        def vector_field(t, y, args):
            z = args
            return dynamic_fn(t, y, z)
        
        term = dfx.ODETerm(vector_field)
        y0 = (x_1, jnp.zeros(1))

        if step_size is not None:
            sol = dfx.diffeqsolve(
                terms=term,
                solver=self.solver,
                t0=1.0,
                t1=0.0,
                dt0= - jnp.fabs(step_size),
                y0=y0,
                saveat=saveat,
                args=args
                )
        else:
            stepsize_controller = dfx.PIDController(atol=atol,rtol=rtol)
            sol = dfx.diffeqsolve(
                terms=term,
                solver=self.solver,
                t0=1.0,
                t1=0.0,
                dt0=None,
                stepsize_controller=stepsize_controller,
                y0=y0,
                saveat=saveat,
                args=args
                )
        x0 = sol.ys[0]
        logp0 = sol.ys[1]
        return x0, logp0
    
    
    def compute_likelihood(
        self,
        velocity_model,
        x_1: Array,
        log_p0: Callable,
        step_size: float | None,
        timegrid = jnp.array([1., 0.]),
        return_intermediates = False,
        exact_divergence: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        enable_grad: bool = False,
        *,
        key
        ):
        # We solve the ODE from 1 to 0
        assert timegrid[0] == 1. and timegrid[-1] == 0.
        
        if return_intermediates:
            saveat = dfx.SaveAt(ts=timegrid)
        else:
            saveat = dfx.SaveAt(t1=True)

        batch_size = x_1.shape[0]
        keys = jxr.split(key, batch_size)

        def dynamics_fn(t, y, z):
            xt, logp = y
            # assert xt.shape == (2 ,), f"shape is {xt.shape}"
            ut = lambda x: velocity_model(x, t)
            if exact_divergence:
                # compute the exact divergence
                div = self._div_jacobian(ut, xt)
            else:
                div = self._div_hutchinson(ut, xt, z)
            # if not enable_grad:
            #     ut = jax.lax.stop_gradient(ut)
            #     div = jax.lax.stop_gradient(div)
            div = jnp.atleast_1d(div)
            dy = ut(xt), div
            return dy

        compute_likelihood_fn = ft.partial(self._compute_likelihood_fn, dynamics_fn, exact_divergence, step_size, atol, rtol, saveat)

        x_traj, log_det_traj = jax.vmap(compute_likelihood_fn)(x_1, keys)
        x_traj = rearrange(x_traj, "b t c -> t b c")
        log_det_traj = rearrange(log_det_traj, "b t 1 -> t b")

        x_source = x_traj[-1]
        log_det_final = log_det_traj[-1]

        source_log_p = jax.vmap(log_p0)(x_source)
        log_p1 = source_log_p + log_det_final
        
        if return_intermediates:
            return x_traj, log_p1
        else:
            return x_source, log_p1
    

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