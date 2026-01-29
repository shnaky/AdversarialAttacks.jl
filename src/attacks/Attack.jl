"""
Abstract supertype for all adversarial attacks.

Expected interface (to be implemented per concrete attack):
- `name(::AbstractAttack)::String`
- `attack(atk::AbstractAttack, model, sample; kwargs...)` returning an adversarial example
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
    attack(atk::AbstractAttack, model, sample; detailed_result=false, kwargs...)

Generate an adversarial example by applying the attack to a sample.

# Arguments
- `atk::AbstractAttack`: Attack configuration and algorithm
- `model`: Target model to attack
- `sample`: Input sample to perturb (e.g., image, text)
- `detailed_result::Bool=false`: 
  - `false` (default): Returns adversarial example only (backward compatible)
  - `true`: Returns `AttackResult` with metrics
- `kwargs...`: Additional attack-specific parameters

# Returns
- `detailed_result=false`: adversarial example only
- `detailed_result=true`: NamedTuple with fields:
  - `x_adv`: Adversarial example
  - `success`: Attack succeeded
  - `queries_used`: Number of model queries
  - `final_label`: Final prediction

"""
function attack(atk::AbstractAttack, model, sample; detailed_result::Bool = false, kwargs...)
    throw(MethodError(attack, (atk, model, sample)))
end
