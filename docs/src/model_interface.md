# Model Interface

```@meta
CurrentModule = AdversarialAttacks
```

```@docs
AbstractModel
DifferentiableModel
NonDifferentiableModel
FluxModel
load_pretrained_c10_model()
AdversarialAttacks.name(::AbstractModel)
AdversarialAttacks.name(::FluxModel)
AdversarialAttacks.predict
AdversarialAttacks.loss
AdversarialAttacks.params
```