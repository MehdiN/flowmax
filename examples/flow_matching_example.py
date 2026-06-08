
import jax
import functools as ft
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jxr
import equinox as eqx
import diffrax as dfx
import optax
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

class Flow(eqx.Module):
    net: eqx.nn.Sequential

    def __init__(self, dim: int=2, h: int=62,* ,key):
        keys = jxr.split(key,4)
        self.net = eqx.nn.Sequential(
            [
                eqx.nn.Linear(dim+1,h,key=keys[0]),
                eqx.nn.Lambda(jnn.elu),
                eqx.nn.Linear(h,h,key=keys[1]),
                eqx.nn.Lambda(jnn.elu),
                eqx.nn.Linear(h,h,key=keys[2]),
                eqx.nn.Lambda(jnn.elu),
                eqx.nn.Linear(h,dim,key=keys[3])
            ]
            )

    def __call__(self, x_t, t):
        t = jnp.atleast_1d(t)
        return self.net(jnp.concat([x_t,t]))


def single_loss_fn(model, x_1, key):
    xkey, tkey = jxr.split(key,2)
    x_0 = jxr.normal(xkey,shape=(2,))
    t = jxr.uniform(tkey)
    x_t = (1-t) * x_0 + t * x_1
    dx_t = x_1 - x_0
    preds = model(x_t, t)
    return optax.l2_loss(preds,dx_t)

def compute_loss(model, x_1, key):
    keys = jxr.split(key,len(x_1))
    loss_fn = ft.partial(single_loss_fn,model)
    loss = jax.vmap(loss_fn)(x_1,keys)
    return jnp.mean(loss)

@eqx.filter_jit
def make_step(model, x_1, key, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(compute_loss)
    loss, grads = loss_fn(model,x_1, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jxr.split(key)[0]
    return loss, model, key, opt_state

@eqx.filter_jit
def single_step_fn(model, x_t, t_0, t_1):
    t_2 = (t_1 - t_0)/2
    return x_t +(t_1 - t_0) * model(x_t + model(x_t, t_0) * t_2, t_0 + t_2)

@eqx.filter_jit
def single_sample_fn(model,dt0,key):
    def flow(t,x, args):
        return model(x,t)
    term = dfx.ODETerm(flow)
    solver = dfx.Tsit5()
    x0=jxr.normal(key,(2,))
    # jax.debug.breakpoint()
    sol = dfx.diffeqsolve(term,solver,t0=0,t1=1,dt0=dt0,y0=x0)
    return sol.ys[0]

def main():

    key = jxr.key(42)
    model = Flow(key=key)
    opt = optax.adam(0.01)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    
    total_loss = 0
    n = 0
    for step in range(10001):
        x_1 = make_moons(256,noise=0.05)[0]
        x_1 = jnp.asarray(x_1, jnp.float32)
        loss_val, model, key, opt_state = make_step(model, x_1, key, opt_state, opt.update)
        total_loss += loss_val
        n+=1
        if step % 100 == 0:
            # jax.debug.breakpoint()
            print(f"Loss={total_loss/n}")
            n = 0
            total_loss = 0

    key = jxr.key(67)
    key, subkey = jxr.split(key)
    x = jxr.normal(key,(300,2))
    n_steps = 8
    timesteps = jnp.linspace(0,1.0,n_steps+1)

    # fig, ax = plt.subplots()
    # xa = make_moons(256,noise=0.05)[0]
    # ax.scatter(xa[:,0],xa[:,1])

    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)

    axes[0].scatter(x[:, 0], x[:, 1], s=10)
    axes[0].set_title(f't = {timesteps[0]:.2f}')
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)
    
    
    for i in range(n_steps):
        t0 = timesteps[i]
        t1 = timesteps[i+1]
        step_fn = ft.partial(single_step_fn,model,t_0=t0,t_1=t1)
        x = jax.vmap(step_fn)(x)
        # jax.debug.breakpoint()
        axes[i + 1].scatter(x[:, 0], x[:, 1], s=10)
        axes[i + 1].set_title(f't = {timesteps[i + 1]:.2f}')

    fig,ax2= plt.subplots()
    num_samples = 300
    sample_keys = jxr.split(subkey,num_samples)
    data_shape = (num_samples, 2)
    sample_fn = ft.partial(single_sample_fn,model,0.1)
    samples = jax.vmap(sample_fn)(sample_keys)
    ax2.scatter(samples[:, 0], samples[:, 1], s=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()