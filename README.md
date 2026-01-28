# AdversarialAttacks.jl
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shnaky.github.io/AdversarialAttacks.jl/dev/)
[![Build Status](https://github.com/shnaky/AdversarialAttacks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shnaky/AdversarialAttacks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/shnaky/AdversarialAttacks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/shnaky/AdversarialAttacks.jl)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AdversarialAttacks.jl** is a lightweight Julia package for experimenting with adversarial attacks against neural networks and tree‑based models, focusing on FGSM (white‑box) and random‑search–based (black‑box) attacks.

Currently, this package supports only models implemented as Flux chains (neural networks) and decision trees from the DecisionTree.jl package. Support for other model types may be added in the future.

## Installation

You can install the package via the Julia package manager.
In the Julia REPL, run:

```julia-repl
julia> ]add https://github.com/shnaky/AdversarialAttacks.jl
```

## Examples

The following example shows how to create an adversarial sample from a **single input sample** using the FGSM attack:

### FGSM attack - Flux Integration (white-box)

```julia-repl
julia> using AdversarialAttacks

julia> using Flux

julia> model = Chain(
           Dense(2, 2, tanh),
           Dense(2, 2),
           softmax,
       )

julia> fgsm = FGSM(; epsilon=0.3)

julia> sample = (data=rand(Float32, 2, 1), label=Flux.onehot(1, 1:2) )

julia> adv_sample = attack(fgsm, model, sample)
```

### BasicRandomSearch attack - DecisionTree integration (black-box)

```julia-repl
julia> using AdversarialAttacks

julia> using DecisionTree

julia> classes = [1, 2, 3]

julia> X = rand(24, 4) .* 4

julia> y = vcat(
    fill(classes[1], 8),
    fill(classes[2], 8),
    fill(classes[3], 8),
)

julia> tree = DecisionTreeClassifier(; classes = classes)

julia> fit!(tree, X, y)

julia> sample = (data = X[:, 1], label = y[1])

julia> atk = BasicRandomSearch(epsilon = 0.1f0, max_iter = 50)

julia> x_adv = attack(atk, tree, sample)
```

### Batch Example
You can also apply an attack to a **batch of samples** represented as a tensor.

```julia-repl
julia> using AdversarialAttacks

julia> using Flux

julia> model = Chain(Dense(4, 8, relu), Dense(8, 2), softmax)

julia> fgsm = FGSM(; epsilon=0.3)

julia> X = rand(Float32, 4, 3)  # 3 samples, 4 features each

julia> Y = Flux.onehotbatch([1, 2, 1], 1:2)  # labels for each sample

julia> tensor = (data=X, label=Y)

julia> adv_samples = attack(fgsm, model, tensor)
```

### Evaluation Example
Get an evaluation report on your adversarial attack.

```julia-repl
julia> using AdversarialAttacks

julia> using Flux

julia> model = Chain(Dense(4, 3), softmax)

julia> test_data = [ (data=randn(Float32, 4), label=Flux.onehot(rand(1:3), 1:3)) for _ in 1:10 ]

julia> fgsm = FGSM(epsilon=0.5)

julia> report = evaluate_robustness(model, fgsm, test_data)

julia> println(report)
=== Robustness Evaluation Report ===

Dataset
  Total samples evaluated        : 10
  Clean-correct samples          : 2 / 10

Clean Performance
  Clean accuracy                 : 20.0%

Adversarial Performance
  Adversarial accuracy           : 50.0%

Attack Effectiveness
  Successful attacks             : 1 / 2
  Attack success rate (ASR)      : 50.0%
  Robustness score (1 - ASR)     : 50.0%

Perturbation Analysis (Norms)
  L_inf Maximum perturbation     : 0.5
  L_inf Mean perturbation        : 0.5
  L_2 Maximum perturbation       : 1.0
  L_2 Mean perturbation          : 1.0
  L_1 Maximum perturbation       : 2.0
  L_1 Mean perturbation          : 2.0

Notes
  • Attack success is counted only when:
    - the clean prediction is correct
    - the adversarial prediction is incorrect
===================================
```

## License
This package is licensed under the MIT License. See [LICENSE](https://github.com/shnaky/AdversarialAttacks.jl/blob/main/LICENSE) for details.

## AI Assistance

This project uses AI coding assistants for routine maintenance tasks such as:
- Code style fixes and documentation improvements
- Consolidating imports and minor refactoring
- Addressing code review feedback

All AI-generated changes are reviewed before merging.
