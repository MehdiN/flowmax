import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jxr
import functools as ft
import equinox as eqx
import equinox.nn as enn
import diffrax as dfx
import optax
import time

# flow matching utilities
from flowmax.path.scheduler import CondOTScheduler
from flowmax.path import AffineProbPath
from flowmax.solvers import ODESolver

import matplotlib.pyplot as plt
from matplotlib import cm

# Generate a checkboard dataset
def inf_train_gen(size: int = 200,*,key):
    key1,key21,key22 = jxr.split(key,3)
    x1 = jxr.uniform(key1, (size, )) * 4 - 2
    tmp_x2 = jxr.uniform(key21, (size, )) - jxr.randint(key22,(size, ),minval=0, maxval=2) * 2.0
    x2 = tmp_x2 + jnp.floor(x1) % 2

    data = 1.0 * jnp.concat([ x1[...,None],x2[...,None]], axis=1) / 0.45

    return jnp.astype(data,jnp.float32)

# Simple MLP for our Flow model
class MLP(eqx.Module):
    net: enn.Sequential

    def __init__(self, dim=2, time_dim=1, hidden_dim=128, *, key) -> None:
        super().__init__()
        keys = jxr.split(key,4)

        self.net = enn.Sequential(
            [
                enn.Linear(dim+time_dim,hidden_dim, key=keys[0]),
                enn.Lambda(jnn.swish),
                enn.Linear(hidden_dim, hidden_dim, key=keys[1]),
                enn.Lambda(jnn.swish),
                enn.Linear(hidden_dim, hidden_dim, key=keys[0]),
                enn.Lambda(jnn.swish),
                enn.Linear(hidden_dim, dim, key=keys[0])
            ]
        )

    def __call__(self, x, t):
        t = jnp.atleast_1d(t)
        x = jnp.concat( [x, t], axis=-1)
        return self.net(x)


# ==================
# Training utilities
# In Jax/Equinox fashion,
# we split training and other function into smaller jittable/vmapable 
# ===================

@eqx.filter_jit
def single_loss_fn(model, path ,x_1, key):
    xkey, tkey, pkey = jxr.split(key, 3)
    x_0 = jxr.normal(xkey, x_1.shape)
    t = jxr.uniform(tkey)
    path_sample = path.sample(x_0=x_0, x_1=x_1, t=t,key=pkey)
    x_t = path_sample.x_t
    dx_t = path_sample.dx_t
    pred_dx_t = model(x_t, path_sample.t)
    loss = optax.l2_loss(pred_dx_t, dx_t)
    return loss

def batch_loss_fn(model, path, x_1, key):
    keys = jxr.split(key, len(x_1))
    loss_fn = ft.partial(single_loss_fn, model, path)
    loss = jax.vmap(loss_fn)(x_1, keys)
    return jnp.mean(loss)

@eqx.filter_jit
def make_train_step(model, path, x_1, opt_state, opt_update, key):
    nkey, retkey = jxr.split(key, 2)
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, path, x_1, nkey)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
    

if __name__=="__main__":

    # training parameters
    lr = 0.001
    data_size = 4096
    batch_size = 64
    epochs = 1000
    print_every = 100 
    hidden_dim = 512

    key = jxr.key(42)
    model_key, train_key = jxr.split(key,2)

    model = MLP(dim=2, time_dim=1, hidden_dim=hidden_dim, key=model_key)

    path = AffineProbPath(scheduler=CondOTScheduler())

    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # Main training loop
    for i in range(epochs):
        xkey, batch_key, train_key = jxr.split(train_key, 3)
        x_1 = inf_train_gen(size=data_size, key= xkey)
        batch_idx = jxr.permutation(batch_key,data_size)
        total_loss = 0
        elpased_time_by_step = 0
        start_time = time.time()
        for j in range(0,data_size,batch_size):
            train_key, subkey = jxr.split(train_key,2)
            batch = x_1[j:j+batch_size]
            loss, model, opt_state = make_train_step(model, path, batch, opt_state, opt.update, key=train_key)
            train_key = subkey
            total_loss += loss.item()
            elapsed = time.time() - start_time
            elpased_time_by_step += elapsed
        if (i+1) % print_every == 0:
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.5f} ' 
                .format(i+1, elpased_time_by_step*1000/data_size, total_loss/data_size)) 
            start_time = time.time()


    # Inference
    print("Sample from trained model")

    # we pass our model in inference mode
    velocity_model = enn.inference_mode(model)
    
    # inference parameters
    step_size = 0.05
    data_size = 10000
    t_steps = 10
    timegrid = jnp.linspace(0,1, t_steps)
    
    # we select a solver from diffrax
    solver = dfx.Tsit5()
    ode_solver = ODESolver(backend=solver)
    # our initial data from N(0,I)
    x_init = jxr.normal(key,(data_size, 2))
    # sample from the model
    sol = ode_solver.sample(velocity_model, x_init, step_size, timegrid, return_intermediates=True)

    # Visualize the flow
    fig, axs = plt.subplots(1, 10,figsize=(20,20))
    for i in range(t_steps):
        H= axs[i].hist2d(sol[i,:,0], sol[i,:,1], 300, range=((-5,5), (-5,5)))
        
        cmin = 0.0
        cmax = jnp.quantile(jnp.asarray(H[0]),0.99).item() 
        norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
        _ = axs[i].hist2d(sol[i,:,0], sol[i,:,1], 300, range=((-5,5), (-5,5)), norm=norm)
        
        axs[i].set_aspect('equal')
        axs[i].axis('off')
        axs[i].set_title('t= %.2f' % (timegrid[i]))
        
    fig.tight_layout()
    plt.show()

