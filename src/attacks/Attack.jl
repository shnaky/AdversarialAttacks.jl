module Attack

"""
Abstract supertype for all adversarial attacks.

Expected interface (to be implemented per concrete attack):
- `name(::AbstractAttack)::String`
- `hyperparameters(::AbstractAttack)::Dict{String,Any}`
- `attack(sample, model, atk::AbstractAttack; kwargs...)` returning an adversarial example
"""
abstract type AbstractAttack end

"""Marker abstract type for white-box attacks."""
abstract type WhiteBoxAttack <: AbstractAttack end

"""Marker abstract type for black-box attacks."""
abstract type BlackBoxAttack <: AbstractAttack end

"""Human-readable name for an attack"""
name(atk::AbstractAttack) = string(typeof(atk))

"""Return hyperparameters for an attack"""
hyperparameters(::AbstractAttack) = Dict{String,Any}()

"""Run the attack on a single sample"""
function attack(sample, model, attack::AbstractAttack; kwargs...)  # TODO: Add return type?
    throw(MethodError(attack, (sample, model, attack)))
end

export AbstractAttack, WhiteBoxAttack, BlackBoxAttack, name, hyperparameters, attack

end
