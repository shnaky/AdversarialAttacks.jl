using Flux

"""
    load_pretrained_c10_model() -> FluxModel

Loads the FluxModel state of a model trained on CIFAR-10 from Github and initializes it as a FluxModel.

# Returns
- `FluxModel`: FluxModel instance with pretrained weights for CIFAR-10 classification.
"""
function load_pretrained_c10_model()

    artifact_toml = LazyArtifacts.find_artifacts_toml(".")
    _hash = artifact_hash("cifar10_model", artifact_toml)

    artifact_dir = artifact_path(_hash)
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
    #https://github.com/user-attachments/files/24865130/cifar10_model.tar.gz
    @load model_path model_state
    Flux.loadmodel!(model, model_state)
    return model
end
