import equinox as eqx
from jaxtyping import Array

class ModelWrapper(eqx.Module):
    model: eqx.Module

    def __init__(self, model: eqx.Module):
        super().__init__()
        self.model = model

    
    def __call__(self, x: Array, t: Array, **kwargs: object):
        return self.model(x=t, t=t,**kwargs)