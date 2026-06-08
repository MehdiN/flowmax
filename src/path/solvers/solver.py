from abc import ABC, abstractmethod
from jaxtyping import Array

class Solver(ABC):

    @abstractmethod
    def sample(self, key, x_0: Array | None = None)->Array:
        ...
