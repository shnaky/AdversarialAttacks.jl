# BasicRandomSearch (Black-Box Attack)

```@meta
CurrentModule = AdversarialAttacks
```

This page documents the BasicRandomSearch algorithm (SimBA variant), a black-box adversarial attack that only requires query access to model predictions.

## BasicRandomSearch Implementation

```@docs
BasicRandomSearch
AdversarialAttacks.attack(::BasicRandomSearch, ::Chain, ::Any)
AdversarialAttacks.attack(::BasicRandomSearch, ::DecisionTreeClassifier, ::Any)
```

## Quick Example

```@example bsr
using AdversarialAttacks

atk = BasicRandomSearch(
    epsilon = 0.3f0,
    bounds = [(0.0f0, 1.0f0), (0.0f0, 1.0f0)],
    max_iter = 50,
)
println("Attack: ", name(atk))
println("Type check: ", atk isa BlackBoxAttack)
println("Epsilon: ", atk.epsilon, ", Max iter: ", atk.max_iter)
```
