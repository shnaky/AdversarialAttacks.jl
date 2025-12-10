module Model

"""
Abstract base for all models that can be attacked.

Must implement: `predict(model, x)`, `loss(model, x, y)`.
Optional: `params(model)` for white-box attacks.
"""
abstract type AbstractModel end

"""
Models that support gradient-based white-box attacks (e.g. neural networks with Flux.jl).
"""
abstract type DifferentiableModel <: AbstractModel end

"""
Models without gradient access for black-box attacks (e.g. traditional ML models such as decision tree, SVM, logistic regression).
"""
abstract type NonDifferentiableModel <: AbstractModel end

"""
    name(m::AbstractModel) -> String

Return a human-readable name for the model.

# Arguments
- `m::AbstractModel`: Model instance.

# Returns
- `String`: Descriptive name of the model type (e.g. `"FluxModel"`, `"TreeModel"`).

# Examples
name(FluxModel(chain)) # "FluxModel"
"""
name(m::AbstractModel) = string(typeof(m))

"""
    predict(m::AbstractModel, x) -> y

Compute model predictions for input `x`.

For classifiers, this usually returns logits or class probabilities.

# Arguments
- `m::AbstractModel`: Model instance.
- `x`: Input data (e.g. feature vector, image batch).

# Returns
- `y`: Model output with a type and shape defined by the concrete model, usually matching the shape expected by `loss`.

# Examples
ŷ = predict(model, x_batch)
"""
predict(m::AbstractModel, x) =
  error("predict not implemented for $(typeof(m))")

"""
    loss(m::AbstractModel, x, y) -> Real

Compute a scalar loss for input–target pair `(x, y)`.

White-box attacks will typically differentiate this loss with respect to the input `x`.

# Arguments
- `m::AbstractModel`: Model instance.
- `x`: Input data.
- `y`: Target labels (e.g. one-hot vectors or class indices).

# Returns
- `Real`: Scalar loss value used for training or attacks.

# Examples
ℓ = loss(model, x_batch, y_batch)
"""
loss(m::AbstractModel, x, y) =
  error("loss not implemented for $(typeof(m))")

"""
    params(m::AbstractModel)

Return the trainable parameters of `m`, if available.

White-box attacks may use this; black-box models can ignore it.

# Arguments
- `m::AbstractModel`: Model instance.

# Returns
- Implementation-defined: Typically a collection of arrays or a parameter container (e.g. `Flux.Params` for `FluxModel`).

# Examples
θ = params(model)
"""
params(m::AbstractModel) =
  error("params not implemented for $(typeof(m))")

export AbstractModel, DifferentiableModel, NonDifferentiableModel, predict, loss, params

end
