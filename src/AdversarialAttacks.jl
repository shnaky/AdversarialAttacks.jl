module AdversarialAttacks

include("attacks/Attack.jl")
include("models/Model.jl")

using .Attack
using .Model

export AbstractAttack, WhiteBoxAttack, BlackBoxAttack, name, hyperparameters, craft

export AbstractModel, DifferentiableModel, NonDifferentiableModel, name, predict, loss, params

end
