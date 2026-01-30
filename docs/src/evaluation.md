# Robustness Evaluation Suite

```@meta 
    CurrentModule = AdversarialAttacks
```
This page documents Robustness Evaluation Suite.

```@docs
RobustnessReport
AdversarialAttacks.calculate_metrics(n_test, num_clean_correct, num_adv_correct, num_successful_attacks, l_norms)
AdversarialAttacks.compute_norm(sample_data, adv_data, p::Real)
AdversarialAttacks.evaluate_robustness(model,attack,test_data;num_samples::Int=100)
AdversarialAttacks.evaluation_curve(model, atk_type::Type{<:AbstractAttack}, epsilons::Vector{Float64}, test_data; num_samples::Int = 100)
```

```@example evalex
using AdversarialAttacks
println("RobustnessReport fields: ", fieldnames(RobustnessReport))
```
