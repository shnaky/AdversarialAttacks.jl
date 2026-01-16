"""
    attack(atk, model, sample; kwargs...)

Apply an adversarial attack to a sample using the given model.

# Arguments
- `atk::AbstractAttack`: The attack object to apply.
- `model::AbstractModel`: The model to attack.
- `sample::AbstractArray{<:Number}` or `NamedTuple`: Input sample (array) or named tuple with `data` and `label` fields.
- `kwargs...`: Additional keyword arguments.

# Returns
- Adversarial sample produced by the attack.

# Notes
- `WhiteBoxAttack` requires a `DifferentiableModel`.
- `BlackBoxAttack` works for any `AbstractModel`.
"""
function attack(atk::WhiteBoxAttack, model::DifferentiableModel, sample::NamedTuple; kwargs...)
    craft(sample, model, atk; kwargs...)
end

function attack(atk::BlackBoxAttack, model::AbstractModel, sample::AbstractArray{<:Number}; kwargs...)
    craft(sample, model, atk; kwargs...)
end

function attack(atk::BlackBoxAttack, model::AbstractModel, sample::NamedTuple; kwargs...)
    craft(sample, model, atk; kwargs...)
end

function attack(atk::WhiteBoxAttack, model::NonDifferentiableModel, sample::AbstractArray{<:Number}; kwargs...)
    error("$(typeof(atk)) is a white-box attack and requires a DifferentiableModel, " *
          "but got $(typeof(model)). Consider using a black-box attack instead.")
end

# for custom attacks that don't subtype WhiteBox/BlackBox
function attack(atk::AbstractAttack, model::AbstractModel, sample::AbstractArray{<:Number}; kwargs...)
    craft(sample, model, atk; kwargs...)
end

"""
    benchmark(atk::AbstractAttack, model::AbstractModel, dataset, metric::Function)

Evaluate attack performance on a dataset with labels using a given metric.

# Arguments
- `atk::AbstractAttack`: Attack algorithm
- `model::AbstractModel`: Target model to attack
- `dataset`: Dataset with samples and labels
- `metric::Function`: Evaluation metric with signature `metric(model, adv_samples, labels)`

# Returns
- Scalar metric value representing attack performance
"""
function benchmark(atk::AbstractAttack, model::AbstractModel, dataset, metric::Function; kwargs...)
    adv_samples = [attack(atk, model, x; kwargs...) for (x, _) in dataset]
    labels = [y for (_, y) in dataset]
    return metric(model, adv_samples, labels)
end
