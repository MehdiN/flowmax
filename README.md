<div align="center">

# Flowmax

**Flow Matching with JAX and Equinox**


[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-orange.svg)]()

</div>

`flowmax` is a [JAX](https://github.com/google/jax)+[Equinox](https://github.com/patrick-kidger/equinox) library for **Flow Matching**, based on the [Flow Matching Guide and Codebase](https://arxiv.org/abs/2412.06264).


The official [PyTorch library](https://github.com/facebookresearch/flow_matching) by Facebook Research can be found [here](https://github.com/facebookresearch/flow_matching).



## Installation

### Dependencies

- `python >= 3.12`
- `jax >= 0.9.2`
- `equinox` – Neural network library
- `diffrax` – ODE solvers
- `optax` – Optimization library
- `einops` – Tensor operations
- `jaxtyping` – Type checking for JAX

### From Source

```bash
git clone https://github.com/MehdiN/flowmax.git
cd flowmax
pip install -e .
```

---

## Quick Start



For how to use the library, see the [`examples/`](examples/) directory.

---

## Module Overview

| Module | Description |
|--------|-------------|
| `flowmax.path/` | Probability path implementations (affine, geodesic, etc.) |
| `flowmax.path.scheduler/` | Scheduling strategies for conditional flow matching |
| `flowmax.solvers/` | ODE solvers for sampling and likelihood computation |
| `flowmax.utils/` | Utility functions |

---

## TODO

- [x] Add likelihood computation
- [ ] Add documentation
- [ ] Add Jupyter notebook examples
- [x] Variance Preserving Schedulers
- [ ] Geodesic and Mixture Probability Paths
- [ ] Solvers for non-Euclidean spaces
- [ ] Discrete Flow Matching
- [ ] Optimize JAX and Equinox routines



## References



```bibtex
@misc{lipman2024flowmatchingguidecode,
      title={Flow Matching Guide and Code},
      author={Yaron Lipman and Marton Havasi and Peter Holderrieth and Neta Shaul and Matt Le and Brian Karrer and Ricky T. Q. Chen and David Lopez-Paz and Heli Ben-Hamu and Itai Gat},
      year={2024},
      eprint={2412.06264},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.06264},
}
```

---

## Acknowledgements

- Thanks to [@facebookresearch](https://github.com/facebookresearch) for the original PyTorch flow matching library.
- Thanks to [@patrick-kidger](https://github.com/patrick-kidger) for [Equinox](https://github.com/patrick-kidger/equinox), [diffrax](https://github.com/patrick-kidger/diffrax), and his ongoing contributions to the JAX ecosystem.