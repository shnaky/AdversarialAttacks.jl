# Attack Interface

```@meta
CurrentModule = AdversarialAttacks
```

This page documents the shared attack abstractions.

```@docs
AbstractAttack
WhiteBoxAttack
BlackBoxAttack
AdversarialAttacks.name(::AbstractAttack)
```

## Type Hierarchy

```@example typehier
using AdversarialAttacks

println("FGSM <: WhiteBoxAttack:       ", FGSM <: WhiteBoxAttack)
println("WhiteBoxAttack <: AbstractAttack: ", WhiteBoxAttack <: AbstractAttack)
println("BasicRandomSearch <: BlackBoxAttack: ", BasicRandomSearch <: BlackBoxAttack)
println("BlackBoxAttack <: AbstractAttack:    ", BlackBoxAttack <: AbstractAttack)
```
