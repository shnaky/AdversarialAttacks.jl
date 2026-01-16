"""
Abstract supertype for all adversarial attacks.

Expected interface (to be implemented per concrete attack):
- `name(::AbstractAttack)::String`
- `hyperparameters(::AbstractAttack)::Dict{String,Any}`
- `craft(sample, model, atk::AbstractAttack; kwargs...)` returning an adversarial example
"""
abstract type AbstractAttack end

"""
Abstract type for white-box adversarial attacks.

White-box attacks have full access to the model's internals, including gradients,
weights, and architecture. This enables the use of gradient-based optimization
and other techniques to craft adversarial examples.

Use this type when the attacker can inspect and manipulate the model's internal
parameters and computations. If only input-output access is available, use
`BlackBoxAttack` instead.
"""
abstract type WhiteBoxAttack <: AbstractAttack end

"""
Abstract type for black-box adversarial attacks.

Black-box attacks only have access to the model's input-output behavior, without knowledge of the model's internals, gradients, or architecture.
These attacks typically rely on query-based methods (e.g., optimization via repeated queries) or transferability from surrogate models.

Use `BlackBoxAttack` when you do not have access to the model's internal parameters or gradients, such as in deployed systems or APIs. 
In contrast, use `WhiteBoxAttack` when you have full access to the model's internals and can leverage gradient information for crafting adversarial examples.
"""
abstract type BlackBoxAttack <: AbstractAttack end

"""
    name(atk::AbstractAttack) -> String

Human-readable name for an attack.

# Returns
- `String`: String representation of the attack type.
"""
name(atk::AbstractAttack)::String = string(typeof(atk))

"""
    hyperparameters(atk::AbstractAttack) -> Dict{String,Any}

Return hyperparameters for an attack.

# Returns
- `Dict{String,Any}`: Dictionary of attack hyperparameters.
"""
hyperparameters(::AbstractAttack)::Dict{String,Any} = Dict{String,Any}()

"""
    craft(sample, model, attack::AbstractAttack; kwargs...) -> adversarial_sample

Craft an adversarial example by applying the attack to a sample.

# Arguments
- `sample`: Input sample to perturb (e.g., image, text)
- `model`: Target model to attack
- `attack::AbstractAttack`: Attack configuration and algorithm
- `kwargs...`: Additional attack-specific parameters

# Returns
- Adversarial example with the same shape as the input sample
"""
function craft(sample, model, attack::AbstractAttack; kwargs...)
    throw(MethodError(craft, (sample, model, attack)))
end
