module AdversarialAttacks

include("attacks/Attack.jl")
include("models/Model.jl")
include("models/FluxModels.jl")

using .Attack
using .Model
using .FluxModels

export AbstractAttack, WhiteBoxAttack, BlackBoxAttack, name, hyperparameters, craft

export AbstractModel, DifferentiableModel, NonDifferentiableModel, FluxModel, predict, loss, params

end
