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


"""
    load_pretrained_c10_model() -> FluxModel

Loads the FluxModel state of a model trained on CIFAR-10 from Github and initializes it as a FluxModel.

# Returns
- `FluxModel`: FluxModel instance with pretrained weights for CIFAR-10 classification.
"""
function load_pretrained_c10_model()

    #url = "https://raw.githubusercontent.com/shnaky/AdversarialAttacks.jl/46-find-a-pretrained-model-for-cifar10-or-build-one/model_cifar10.bson"
    #MODEL_PATH = joinpath(@__DIR__, "..", "models", "model_cifar10.bson")
    #print(MODEL_PATH)

    artifact_dir = artifact"cifar10_model"
    model_path = joinpath(artifact_dir, "model_cifar10.bson")
    

    model = Chain(

        # Block 1
        Conv((3,3), 3 => 96, pad=1),
        relu,
        Dropout(0.2),

        Conv((3,3), 96 => 96, pad=1),
        relu,

        Conv((3,3), 96 => 96, pad=1),
        relu,

        MaxPool((3,3), stride=2),
        Dropout(0.5),

        # Block 2
        Conv((3,3), 96 => 192, pad=1),
        relu,

        Conv((3,3), 192 => 192, pad=0),   
        relu,

        Conv((3,3), 192 => 192, pad=1),
        relu,

        MaxPool((3,3), stride=2),
        Dropout(0.5),

        # Block 3
        Conv((3,3), 192 => 192, pad=1),
        relu,

        Conv((1,1), 192 => 192, pad=0),
        relu,

        Conv((1,1), 192 => 10, pad=0),

        GlobalMeanPool(),
        Flux.flatten
    )
    #@load download(url, "model_cifar10_downloaded.bson") model_state
    @load model_path model_state
    Flux.loadmodel!(model, model_state)
    return FluxModel(model)
end
