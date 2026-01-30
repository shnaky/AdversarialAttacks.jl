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

# Construct attacks via the high-level API types
fgsm = FGSM(epsilon = 0.01f0)
bsr = BasicRandomSearch(epsilon = 0.1f0, bounds = [(0f0, 1f0)], max_iter = 10)

println("FGSM: ", name(fgsm))
println("BSR:  ", name(bsr))
```
