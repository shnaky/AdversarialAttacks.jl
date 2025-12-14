module Interface

export run, benchmark

using ..Attack: AbstractAttack, craft
using ..Model: AbstractModel

"""
    run(attack, model, sample)

Fallback function that gives an error if no specific method exists

# Throws
- `MethodError`: When no specific method is implemented
"""
function run(attack, model, sample)
    throw(MethodError(run, (attack, model, sample)))
end

"""
    run(attack::AbstractAttack, model::AbstractModel, sample::AbstractArray)

Craft a single adversarial example by applying the attack to input `sample`.

# Arguments
- `attack::AbstractAttack`: Attack algorithm
- `model::AbstractModel`: Target model to attack
- `sample::AbstractArray`: Input sample

# Returns
- Adversarial sample
"""
function run(attack::AbstractAttack, model::AbstractModel, sample::AbstractArray; kwargs...)
    return craft(sample, model, attack; kwargs...)
end

"""
    run(attack::AbstractAttack, model::AbstractModel, samples::AbstractVector{<:AbstractArray})

Batch processing: craft adversarial examples for multiple inputs.

# Arguments
- `attack::AbstractAttack`: Attack algorithm
- `model::AbstractModel`: Target model to attack
- `samples::AbstractVector{<:AbstractArray}`: Vector of input samples

# Returns
- Adversarial samples
"""
function run(attack::AbstractAttack, model::AbstractModel, samples::AbstractVector{<:AbstractArray}; kwargs...)
    return [run(attack, model, sample; kwargs...) for sample in samples]
end

"""
    benchmark(attack::AbstractAttack, model::AbstractModel, dataset::AbstractVector{<:Tuple}, metric)::Number

Evaluate attack performance on a dataset with labels using a given metric

# Arguments
- `attack::AbstractAttack`: Attack algorithm
- `model::AbstractModel`: Target model to attack
- `dataset::AbstractVector{<:Tuple}`: Vector of (input, label) pairs
- `metric::Function`: Evaluation metric

# Returns
- `Number`: represents the metric score
"""
function benchmark(attack::AbstractAttack, model::AbstractModel, dataset::AbstractVector{<:Tuple}, metric::Function; kwargs...)::Number
    adv_samples = [run(attack, model, x; kwargs...) for (x, _) in dataset]
    labels = [y for (_, y) in dataset]
    return metric(adv_samples, labels)
end

end # module
