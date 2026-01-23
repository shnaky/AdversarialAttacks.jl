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
    attack(atk::FGSM, model, sample)

Perform a Fast Gradient Sign Method (FGSM) white-box adversarial attack on the given `model` using the provided `sample`.

# Arguments
- `atk::FGSM`: An instance of the `FGSM`.
- `model::FluxModel`: The machine learning (deep learning) model to be attacked.
- `sample`: Input sample as a named tuple with `data` and `label`.

# Returns
- Adversarial example (same type and shape as `sample.data`).
"""
function attack(atk::FGSM, model::Chain, sample; loss = default_loss)
    x = sample.data
    y = sample.label
    ε = convert(eltype(x), atk.epsilon)
    # Compute gradient of loss w.r.t. input x
    grads = gradient(xx -> loss(model, xx, y), x)[1]
    # FGSM perturbation: ε · sign(∇_x L(x, y))
    perturbation = ε .* sign.(grads)
    adversarial_example = x .+ perturbation
    return adversarial_example
end
