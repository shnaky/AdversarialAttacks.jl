# Tutorials & Examples Overview

The **Tutorials & Examples** section provides end‑to‑end, runnable scripts that show how to use AdversarialAttacks.jl in realistic workflows. Each tutorial is generated from the scripts in `examples/` using [Literate.jl](https://fredrikekre.github.io/Literate.jl/), so the code runs during the documentation build and all output is captured inline.

## The current tutorials cover:

- **White-box FGSM attack** against a small Flux CNN on MNIST — see the [tutorial](../tutorials/whitebox_fgsm_flux_mnist.md).
- **Black-box Basic Random Search** (SimBA-style) attack against a DecisionTree classifier on the Iris dataset — see the [tutorial](../tutorials/blackbox_basicrandomsearch_decisiontree_iris.md).

## Running locally

The tutorial scripts in `examples/` are fully standalone. You can run them directly from the repository root:

```julia
julia> using Pkg
julia> Pkg.activate("./examples")   # activate the examples environment from the repository root
julia> include("examples/whitebox_fgsm_flux_mnist.jl")
julia> include("examples/blackbox_basicrandomsearch_decisiontree_iris.jl")
```

Or run them directly from the `examples/` directory using the project flag:

```
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. whitebox_fgsm_flux_mnist.jl
julia --project=. blackbox_basicrandomsearch_decisiontree_iris.jl
```

## Notes and tips

- Each example uses a small training setup so it runs quickly for demonstration purposes. For a more applicable example, re-run with more epochs and save the trained model.
- Scripts will print training and attack statistics and generate plot files.
- If you want to reproduce results exactly, set seeds as shown in the examples.

## Next steps

- Follow the individual tutorial pages for step‑by‑step code with executed output and plots
