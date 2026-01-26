# Tutorials & Examples Overview

The **Tutorials & Examples** section provides end‑to‑end, runnable scripts that demonstrate how to use AdversarialAttacks.jl in realistic workflows. Each example focuses on a concrete combination of model type, dataset, and attack method, and comes with its own project environment for reproducible execution.

The current examples cover:

- A white‑box FGSM attack against a Flux CNN on MNIST.
- A black‑box Basic Random Search attack against a DecisionTree classifier on the Iris dataset.

When working from the package environment in a Julia REPL started in the repository root, you can activate the dedicated examples environment and run the scripts as follows:

```julia
julia> using Pkg
julia> Pkg.activate("./examples")   # activate the examples environment from the repository root
julia> include("examples/whitebox_fgsm_flux_mnist.jl")
julia> include("examples/blackbox_basicrandomsearch_decisiontree_iris.jl")
```

From the `examples/` directory, all scripts can be run with:

```julia
julia --project=. whitebox_fgsm_flux_mnist.jl
julia --project=. blackbox_basicrandomsearch_decisiontree_iris.jl
```

Each tutorial logs training and attack statistics and opens Plots.jl visualizations to inspect the resulting adversarial examples.