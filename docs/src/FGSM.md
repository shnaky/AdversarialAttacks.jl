# FGSM (White-Box Attack)

```@meta
CurrentModule = AdversarialAttacks
```

This page documents the [Fast Gradient Sign Method (FGSM)](https://arxiv.org/abs/1412.6572), a White-box Adversarial Attack that requires access to model gradients.

## FGSM Implementation

```@docs
FGSM
AdversarialAttacks.attack(::FGSM, ::Chain, ::Any)
```

## Quick Example

```@example fgsm
using AdversarialAttacks

atk = FGSM(epsilon = 0.01f0)
println("Attack: ", name(atk))
println("Type check: ", atk isa WhiteBoxAttack)
println("Epsilon: ", atk.epsilon)
```
