from abc import ABC, abstractmethod
import jax
from .scheduler import Scheduler
from .path_sample import PathSample

@jax.tree_util.register_pytree_node_class
class BaseProbPath(ABC):
    scheduler: Scheduler

    """
    Intended usage

    for x_1, x_0 in dataset:
        t = jax.random.uniform(...)
        path_sample = my_path.sample(x_0,x_1,t,key)

        ## optimize

    """
    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    @abstractmethod
    def sample(self,x_0:jax.Array, x_1:jax.Array, t: jax.Array, *, key) -> PathSample:
        """
           TODO: add doc
        """
    ...

    # to register the path as a Pytree
    def tree_flatten(self):
        children = None
        aux_data = {"scheduler": self.scheduler}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, chidren):
        return cls(*chidren, **aux_data)


    # not useful for jax code ?
    # def assert_sample_shape(self, x_0: jax.Array, x_1: jax.Array, t: jax.Array):
    #     assert (
    #         t.ndim == 1
    #     )
    #     assert (
    #         t.shape[0] == x_0.shape[0] == x_1.shape[0]
    #     )