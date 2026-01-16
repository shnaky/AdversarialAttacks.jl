using Flux

"""
    FluxModel(model::Flux.Chain)

Flux-based differentiable model wrapper.

Allows using `Flux.Chain` with the `DifferentiableModel` interface.

# Arguments
- `model::Flux.Chain`: Flux chain model to wrap.

# Examples
```
chain = Chain(Dense(10 => 5), Dense(5 => 2))
model = FluxModel(chain)
```
"""
struct FluxModel <: DifferentiableModel
    model::Flux.Chain
end


"""
    name(m::FluxModel)

Return a human-readable name for the Flux model.

# Returns
- `String`: `"FluxModel"`
"""
name(::FluxModel)::String = "FluxModel"



"""
    predict(m::FluxModel, x)

Forward pass: delegate to the wrapped `Flux.Chain`.

# Arguments
- `m::FluxModel`: FluxModel instance
- `x`: input data

# Returns
- `model(x)`: Flux chain output
"""
predict(m::FluxModel, x) = m.model(x)



"""
    loss(m::FluxModel, x, y)

Cross-entropy loss for classification tasks.

# Arguments
- `m::FluxModel`: FluxModel instance
- `x`: input data
- `y`: ground truth labels (one-hot or class indices)


# Returns
- `Float32`: logitcrossentropy loss value
"""
loss(m::FluxModel, x, y) = Flux.logitcrossentropy(m.model(x), y)


"""
    params(m::FluxModel)

Return all trainable parameters of the wrapped Flux model.
White-box attacks may use this; black-box models can ignore it.

# Arguments
- `m::FluxModel`: FluxModel instance

# Returns
- `Flux.Params`: collection of trainable parameters

# Examples
```
θ = params(model)
grads = gradient(θ) do
    loss(model, x, y)
end
```
"""
params(m::FluxModel) = Flux.trainable(m.model)
