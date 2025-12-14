# AdversarialAttacks.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://shnaky.github.io/AdversarialAttacks.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shnaky.github.io/AdversarialAttacks.jl/dev/)
[![Build Status](https://github.com/shnaky/AdversarialAttacks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shnaky/AdversarialAttacks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/shnaky/AdversarialAttacks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/shnaky/AdversarialAttacks.jl)

## Installation

You can install the package via the Julia package manager.
In the Julia REPL, run:

```julia-repl
julia> ]add AdversarialAttacks
```

## Examples
The following example is an example to create an adversarial sample out of a singe sample using the FGSM attack algorithm.

```julia-repl
julia> using AdversarialAttacks

julia> struct MyModel <: DifferentiableModel end

julia> fgsm = FGSM(ε=0.3)

julia> model = MyModel()

julia> sample = rand(28, 28)

julia> adversarial_sample = AdversarialAttacks.run(fgsm, model, sample)
```

### Batch Example

```julia-repl
julia> samples = [rand(28, 28) for _ in 1:10]

julia> adversarial_samples = AdversarialAttacks.run(fgsm, model, samples)
```

### Benchmark
> **TODO**
