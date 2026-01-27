# Tutorials & Examples Overview

The **Tutorials & Examples** section provides end‑to‑end, runnable scripts that show how to use AdversarialAttacks.jl in realistic workflows. Each example pairs a dataset and model with a concrete attack method and includes a compact, reproducible script you can run locally.
All examples are thought to be a start for your own adventures, and will not be maintained. 

## The current examples cover:

- White-box FGSM attack against a small Flux CNN on MNIST (file: `whitebox_fgsm_flux_mnist`).
- Black-box Basic Random Search (SimBA-style) attack against a DecisionTree classifier on the Iris dataset (file: `blackbox_basicrandomsearch_decisiontree_iris`).

## Quick start

From the repository root you can activate the `examples` environment and run the scripts in a Julia REPL:

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
- Scripts will print training and attack statistics and open Plots.jl visualizations to inspect original vs adversarial examples.
- If you want to reproduce results exactly, set seeds as shown in the examples.

## Next steps

- Follow the individual example pages for step‑by‑step code and explanations