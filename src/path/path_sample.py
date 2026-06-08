from dataclasses import dataclass, field
import jax


@jax.tree_util.register_dataclass
@dataclass
class PathSample:
    """
        Represents a sample of a conditional-flow generated path.
    """
    x_1: jax.Array
    x_0: jax.Array
    t: jax.Array
    x_t: jax.Array
    dx_t: jax.Array

@jax.tree_util.register_dataclass
@dataclass
class DiscretePathSample:
    """
        Represents a sample of a conditional-flow generated discrete probability path
    """
    x_1: jax.Array
    x_0: jax.Array
    t: jax.Array
    x_t: jax.Array

