# Interface

```@meta
CurrentModule = AdversarialAttacks
```

This page documents the Interface for user interaction.

```@docs
AdversarialAttacks.attack
AdversarialAttacks.benchmark
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