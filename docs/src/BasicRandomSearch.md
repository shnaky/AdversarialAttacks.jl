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
