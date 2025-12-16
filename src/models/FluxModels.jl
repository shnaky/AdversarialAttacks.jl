module FluxModels

using ..Model
using Flux

"""
    FluxModel(model::Flux.Chain)

Flux-based differentiable model wrapper.

Allows using `Flux.Chain` with the `Model.DifferentiableModel` interface.

# Arguments
- `model::Flux.Chain`: Flux chain model to wrap.

# Examples
```
chain = Chain(Dense(10 => 5), Dense(5 => 2))
model = FluxModel(chain)
```
"""
struct FluxModel <: Model.DifferentiableModel
  model::Flux.Chain
end


"""
    Model.name(m::FluxModel)

Return a human-readable name for the Flux model.

# Returns
- `String`: `"FluxModel"`
"""
Model.name(::FluxModel) = "FluxModel"



"""
    Model.predict(m::FluxModel, x)

Forward pass: delegate to the wrapped `Flux.Chain`.

# Arguments
- `m::FluxModel`: FluxModel instance
- `x`: input data

# Returns
- `model(x)`: Flux chain output
"""
Model.predict(m::FluxModel, x) = m.model(x)



"""
    Model.loss(m::FluxModel, x, y)

Cross-entropy loss for classification tasks.

# Arguments
- `m::FluxModel`: FluxModel instance
- `x`: input data
- `y`: ground truth labels (one-hot or class indices)


# Returns
- `Float32`: logitcrossentropy loss value
"""
Model.loss(m::FluxModel, x, y) = Flux.logitcrossentropy(m.model(x), y)


"""
    Model.params(m::FluxModel)

Return all trainable parameters of the wrapped Flux model.
White-box attacks may use this; black-box models can ignore it.

# Arguments
- `m::FluxModel`: FluxModel instance

# Returns
- `Flux.Params`: collection of trainable parameters

# Examples
```
θ = Model.params(model)
grads = gradient(θ) do
    Model.loss(model, x, y)
end
```
"""
Model.params(m::FluxModel) = Flux.trainable(m.model)

export FluxModel

end