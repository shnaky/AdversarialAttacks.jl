# FGSM (White-Box Attack)

```@meta
CurrentModule = AdversarialAttacks
```

This page documents the Fast Gradient Sign Method (FGSM), a white-box adversarial attack that requires access to model gradients.

## FGSM Implementation

```@docs
FGSM
AdversarialAttacks.attack(::FGSM, ::Chain, ::Any)
AdversarialAttacks.hyperparameters(::FGSM)
```
