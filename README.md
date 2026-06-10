<div align="center">

# Flowmax

Flow Matching with Jax and Equinox

</div>

`flowmax` is a JAX librairy for Flow Matching and based
on the [Flow Matching Guide et Codebase](https://arxiv.org/abs/2412.06264).
It is a rewrite of the associated [Pytorch library](`https://github.com/facebookresearch/flow_matching`) with Jax and Equinox models in mind.

Dependencies:
- `python >= 3.12`
- `jax >= 0.9.2`
- `equinox` for the neural network library
- `diffrax` for ODE solvers
- `optax` for optimization
- `jaxtyping`

## Usage exemples

check the examples in `examples/`

## TODO

- [ ] Add likelihood computation
- [ ] Add documentation
- [ ] Add more exemples
- [ ] Variance Preserving Schedulers
- [ ] Geodesic and Mixture Probability Paths
- [ ] Solvers for non Euclidian spaces
- [ ] Discrete Flow Matching

## Reference

```
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


## Aknowledgements

Thanks to @facebookresearch for the original library

Thanks to @patrick-kidger for equinox, diffrax and his ongoing
contributions to the Jax ecosystem
