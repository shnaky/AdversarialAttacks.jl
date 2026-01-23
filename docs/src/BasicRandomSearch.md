# BasicRandomSearch (Black-Box Attack)

```@meta
CurrentModule = AdversarialAttacks
```

This page documents the BasicRandomSearch algorithm (SimBA variant), a black-box adversarial attack that only requires query access to model predictions.

## BasicRandomSearch Implementation

```@docs
BasicRandomSearch
AdversarialAttacks.craft(::Any, ::Chain, ::BasicRandomSearch)
AdversarialAttacks.craft(::Any, ::DecisionTreeClassifier, ::BasicRandomSearch)
```
