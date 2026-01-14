module AdversarialAttacks

include("attacks/Attack.jl")
include("models/Model.jl")
include("models/FluxModels.jl")
include("attacks/BlackBoxAttacks.jl")
include("attacks/WhiteBox.jl")
include("Interface.jl")

# Export attack types
export AbstractAttack, WhiteBoxAttack, BlackBoxAttack
export FGSM, BasicRandomSearch, SquareAttack

# Export model types
export AbstractModel, DifferentiableModel, NonDifferentiableModel
export FluxModel

# Export attack interface functions
export craft, name, hyperparameters

# Export model interface functions
export predict, loss, params

# Export high-level interface
export attack, benchmark

end
