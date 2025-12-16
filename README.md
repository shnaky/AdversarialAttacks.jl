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

>[!WARNING]
>This project is still in early development. The attack algorithms have not been fully implemented yet.

The following example shows how to create an adversarial sample from a **single input sample** using the FGSM attack:

```julia-repl
julia> using AdversarialAttacks

julia> struct MyModel <: DifferentiableModel end

julia> fgsm = FGSM(Dict("ε"=>0.3))

julia> model = MyModel()

julia> sample = rand(10, 10)

julia> adv_sample = attack(fgsm, model, sample)
```

### Batch Example
You can also apply an attack to a **batch of samples** represented as a tensor.

```julia-repl
julia> tensor = rand(2, 2, 3)

julia> adv_samples = attack(fgsm, model, tensor)
```

### Flux Integration
This package can also work with **Flux.jl models**. Wrap your Flux model using the `FluxModel` interface:

```julia-repl
julia> using Flux

julia> m = Chain(Dense(2, 2, relu), Dense(2, 2))

julia> model = FluxModel(m)

julia> fgsm = FGSM(Dict("ε"=>0.3))

julia> sample = rand(10, 10)

julia> adv_sample = attack(fgsm, model, sample)
```

