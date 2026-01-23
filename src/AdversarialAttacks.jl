module AdversarialAttacks

using Random: randperm
using Downloads: download
using BSON: @load

include("attacks/Attack.jl")
include("models/Model.jl")
include("models/FluxModels.jl")
include("attacks/BlackBoxAttacks.jl")
include("attacks/WhiteBox.jl")
include("Interface.jl")
include("Evaluation.jl")
include("examples/cifar10.jl")

# Export attack types
export AbstractAttack, WhiteBoxAttack, BlackBoxAttack
export FGSM, BasicRandomSearch, SquareAttack

# Export model types
export AbstractModel, DifferentiableModel, NonDifferentiableModel
export FluxModel
export load_pretrained_c10_model, plot_adversarial_example 

# Export attack interface functions
export craft, name, hyperparameters

# Export model interface functions
export predict, loss, params

# Export evaluation functions
export evaluate_robustness

# Export high-level interface
export attack, benchmark

end
