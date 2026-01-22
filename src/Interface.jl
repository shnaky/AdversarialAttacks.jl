using DecisionTree: DecisionTreeClassifier
using Flux

"""
    attack(atk, model, sample; kwargs...)

Apply an adversarial attack to a sample using the given model.

# Arguments
- `atk::AbstractAttack`: The attack object to apply.
- `model`: The model to attack. Supports:
    * `Flux.Chain` (for white-box and black-box attacks)
    * `DecisionTreeClassifier` (for black-box attacks)
- `sample::AbstractArray{<:Number}` or `NamedTuple`:
    * Raw input array, or
    * NamedTuple with `data` and `label` fields.
- `kwargs...`: Additional keyword arguments.

# Returns
- Adversarial sample produced by the attack.

# Notes
- `WhiteBoxAttack` is supported for `Flux.Chain`.
- `BlackBoxAttack` is supported for both `Flux.Chain` and `DecisionTreeClassifier`
  (treated as black-box models, using only model outputs).
"""
function attack(atk::WhiteBoxAttack, model::Flux.Chain, sample::NamedTuple; kwargs...)
    craft(sample, model, atk; kwargs...)
end

function attack(atk::WhiteBoxAttack, model::Flux.Chain, sample::AbstractArray{<:Number}; kwargs...)
    craft(sample, model, atk; kwargs...)
end

function attack(atk::BlackBoxAttack, model::Flux.Chain, sample::NamedTuple; kwargs...)
    craft(sample, model, atk; kwargs...)
end

function attack(atk::BlackBoxAttack, model::Flux.Chain, sample::AbstractArray{<:Number}; kwargs...)
    craft(sample, model, atk; kwargs...)
end

# for custom attacks that don't subtype WhiteBox/BlackBox
function attack(atk::AbstractAttack, model, sample::AbstractArray{<:Number}; kwargs...)
    craft(sample, model, atk; kwargs...)
end

function attack(atk::BlackBoxAttack, model::DecisionTreeClassifier, sample::NamedTuple; kwargs...)
    craft(sample, model, atk; kwargs...)
end

function attack(atk::BlackBoxAttack, model::DecisionTreeClassifier, sample::AbstractArray{<:Number}; kwargs...)
    craft(sample, model, atk; kwargs...)
end

"""
    benchmark(atk::AbstractAttack, model, dataset, metric::Function; kwargs...)

Evaluate attack performance on a dataset with labels using a given metric.

# Arguments
- `atk::AbstractAttack`: Attack algorithm
- `model`: Target model to attack
- `dataset`: Dataset with samples and labels
- `metric::Function`: Evaluation metric with signature `metric(model, adv_samples, labels)`

# Returns
- Scalar metric value representing attack performance
"""
function benchmark(atk::AbstractAttack, model, dataset, metric::Function; kwargs...)
    adv_samples = [attack(atk, model, x; kwargs...) for (x, _) in dataset]
    labels = [y for (_, y) in dataset]
    return metric(model, adv_samples, labels)
end
