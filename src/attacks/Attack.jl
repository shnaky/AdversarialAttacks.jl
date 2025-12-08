module Attack

"""
Abstract supertype for all adversarial attacks.

Expected interface (to be implemented per concrete attack):
- `name(::AbstractAttack)::String`
- `hyperparameters(::AbstractAttack)::Dict{String,Any}`
- `attack(sample, model, atk::AbstractAttack; kwargs...)` returning an adversarial example
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

"""Human-readable name for an attack"""
name(atk::AbstractAttack) = string(typeof(atk))

"""Return hyperparameters for an attack"""
hyperparameters(::AbstractAttack) = Dict{String,Any}()

"""Run the attack on a single sample"""
function attack(sample, model, attack::AbstractAttack; kwargs...)  # TODO: Add return type?
    throw(MethodError(run, (sample, model, attack)))
end

export AbstractAttack, WhiteBoxAttack, BlackBoxAttack, name, hyperparameters, attack

end
