"""
    FGSM(; epsilon=0.1)

A struct that can be used to create a white-box adversarial attack of type Fast Gradient Sign Method.
Subtype of `WhiteBoxAttack`.

# Arguments
- `epsilon`: Step size used to scale the sign of the gradient. Defaults to `0.1`.
"""
struct FGSM{T <: Real} <: WhiteBoxAttack
    epsilon::T
end

FGSM(; epsilon::Real = 0.1) = FGSM(epsilon)

"""
    hyperparameters(atk::FGSM) -> Dict{String,Any}

Return hyperparameters for an FGSM attack.

# Returns
- `Dict{String,Any}`: Dictionary containing attack hyperparameters (e.g., epsilon).
"""
hyperparameters(atk::FGSM)::Dict{String, Any} = Dict("epsilon" => atk.epsilon)

default_loss(m, x, y) = crossentropy(m(x), y)

"""
    craft(sample, model, attack::FGSM)

Perform a Fast Gradient Sign Method (FGSM) white-box adversarial attack on the given `model` using the provided `sample`.

# Arguments
- `sample`: Input sample as a named tuple with `data` and `label`.
- `model::FluxModel`: The machine learning (deep learning) model to be attacked.
- `attack::FGSM`: An instance of the `FGSM`.

# Returns
- Adversarial example (same type and shape as `sample.data`).
"""
function craft(sample, model::Chain, attack::FGSM; loss = default_loss)
    x = sample.data
    y = sample.label
    ε = convert(eltype(x), attack.epsilon)
    # Compute gradient of loss w.r.t. input x
    grads = gradient(xx -> loss(model, xx, y), x)[1]
    # FGSM perturbation: ε · sign(∇_x L(x, y))
    perturbation = ε .* sign.(grads)
    adversarial_example = x .+ perturbation
    return adversarial_example
end
