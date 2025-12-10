module Interface

export run, benchmark

abstract type AbstractAttack end # placeholder
abstract type AbstractModel end # placeholder

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
"""
# TODO: Differantiate this bwetween AbstractAttack and AbstractModel
function run(attack::AbstractAttack, model::AbstractModel, sample::AbstractArray)
    # TODO: This should be overridden by specific attack implementations
    return sample # placeholder
end

"""
    run(attack::AbstractAttack, model::AbstractModel, samples::AbstractVector{<:AbstractArray})

Batch processing: craft adversarial examples for multiple inputs.

# Arguments
- `attack::AbstractAttack`: Attack algorithm
- `model::AbstractModel`: Target model to attack
- `samples::AbstractVector{<:AbstractArray}`: Vector of input samples
"""
function run(attack::AbstractAttack, model::AbstractModel, samples::AbstractVector{<:AbstractArray})
    return [run(attack, model, sample) for sample in samples]
end

"""
    benchmark(attack::AbstractAttack, model::AbstractModel, dataset::AbstractVector{<:Tuple}, metric)::Number

Evaluate attack performance on a dataset with labels using a given metric

# Arguments
- `attack::AbstractAttack`: Attack algorithm
- `model::AbstractModel`: Target model to attack
- `dataset::AbstractVector{<:Tuple}`: Vector of (input, label) pairs
- `metric`: Evaluation metric
"""
function benchmark(attack::AbstractAttack, model::AbstractModel, dataset::AbstractVector{<:Tuple}, metric)::Number
    return 0.0 # placeholder
end

end # end module