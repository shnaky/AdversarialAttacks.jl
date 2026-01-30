# Attack Interface

```@meta
CurrentModule = AdversarialAttacks
```

This page documents the attack interface for generating adversarial examples.

## API Reference

```@docs
AbstractAttack
WhiteBoxAttack
BlackBoxAttack
AdversarialAttacks.name(::AbstractAttack)
AdversarialAttacks.attack
```

## Type Hierarchy

```@example typehier
using AdversarialAttacks

println("FGSM <: WhiteBoxAttack:       ", FGSM <: WhiteBoxAttack)
println("WhiteBoxAttack <: AbstractAttack: ", WhiteBoxAttack <: AbstractAttack)
println("BasicRandomSearch <: BlackBoxAttack: ", BasicRandomSearch <: BlackBoxAttack)
println("BlackBoxAttack <: AbstractAttack:    ", BlackBoxAttack <: AbstractAttack)
```

## Quick Example

```@example iface
using AdversarialAttacks
using Flux

# Construct attacks via the high-level API types
fgsm = FGSM(epsilon = 0.01f0)
brs = BasicRandomSearch(epsilon = 0.1f0, max_iter = 10)
brs2 = BasicRandomSearch(epsilon = 0.1f0, bounds = [(0f0, 1f0)], max_iter = 10)

println("FGSM: ", name(fgsm))
println("BSR:  ", name(brs))

# Run an attack
model = Chain(
           Dense(2, 2, tanh),
           Dense(2, 2),
           softmax,
       )
sample = (data=rand(Float32, 2, 1), label=Flux.onehot(1, 1:2))

adv_sample_fgsm = attack(fgsm, model, sample)
adv_sample_brs = attack(brs, model, sample)
```
