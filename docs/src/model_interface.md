# Model Interface

```@meta
CurrentModule = AdversarialAttacks
```

```@docs
AbstractModel
DifferentiableModel
NonDifferentiableModel
FluxModel
AdversarialAttacks.name(::AbstractModel)
AdversarialAttacks.name(::FluxModel)
AdversarialAttacks.predict
AdversarialAttacks.loss
AdversarialAttacks.params
```