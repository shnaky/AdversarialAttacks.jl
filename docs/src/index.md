# AdversarialAttacks.jl
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shnaky.github.io/AdversarialAttacks.jl/dev/)
[![Build Status](https://github.com/shnaky/AdversarialAttacks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shnaky/AdversarialAttacks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/shnaky/AdversarialAttacks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/shnaky/AdversarialAttacks.jl)

## Installation

You can install the package via the Julia package manager.
In the Julia REPL, run:

```julia-repl
julia> ]add https://github.com/shnaky/AdversarialAttacks.jl
```

## Examples

>[!WARNING]
>This project is still in early development. The attack algorithms have not been fully implemented yet.

The following example shows how to create an adversarial sample from a **single input sample** using the FGSM attack:

```julia-repl
julia> using AdversarialAttacks

julia> struct MyModel <: DifferentiableModel end

julia> AdversarialAttacks.predict(::MyModel, x) = sigmoid.(sum(x, dims=1))

julia> AdversarialAttacks.loss(::MyModel, x, y) = Flux.binarycrossentropy(predict(MyModel(), x), y)

julia> AdversarialAttacks.params(::MyModel) = Flux.Params([])

julia> model = MyModel()

julia> fgsm = FGSM(; epsilon=0.3)

julia> sample = (data=rand(Float32, 10, 1), label=1)

julia> adv_sample = attack(fgsm, model, sample)
```

### Flux Integration
This package can also work with **Flux.jl models**. Wrap your Flux model using the `FluxModel` interface:

```julia-repl
julia> using AdversarialAttacks

julia> using Flux

julia> m = Chain(Dense(2, 2, tanh), Dense(2, 2))

julia> model = FluxModel(m)

julia> fgsm = FGSM(; epsilon=0.3)

julia> sample = (data=rand(Float32, 2, 1), label=Flux.onehot(1, 1:2) )

julia> adv_sample = attack(fgsm, model, sample)
```

### Batch Example
You can also apply an attack to a **batch of samples** represented as a tensor.

```julia-repl
julia> using AdversarialAttacks

julia> using Flux

julia> m = Chain(Dense(4, 8, relu), Dense(8, 2))

julia> model = FluxModel(m)

julia> fgsm = FGSM(; epsilon=0.3)

julia> X = rand(Float32, 4, 3)  # 3 samples, 4 features each

julia> Y = Flux.onehotbatch([1, 2, 1], 1:2)  # labels for each sample

julia> tensor = (data=X, label=Y)

julia> adv_samples = attack(fgsm, model, tensor)
```