from abc import ABC, abstractmethod
import jax
from .path_sample import PathSample

class BaseProbPath(ABC):

    """
    Intended usage

    for x_1, x_0 in dataset:
        t = jax.random.uniform(...)
        path_sample = my_path.sample(x_0,x_1,t,key)

        ## optimize

    """
    @abstractmethod
    def sample(self,x_0:jax.Array, x_1:jax.Array, t: jax.Array, *, key) -> PathSample:
        """
           TODO: add doc
        """
    ...
    # not useful for jax code ?
    # def assert_sample_shape(self, x_0: jax.Array, x_1: jax.Array, t: jax.Array):
    #     assert (
    #         t.ndim == 1
    #     )
    #     assert (
    #         t.shape[0] == x_0.shape[0] == x_1.shape[0]
    #     )