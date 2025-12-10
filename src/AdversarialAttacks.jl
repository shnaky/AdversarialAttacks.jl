module AdversarialAttacks

include("attacks/Attack.jl")
include("models/Model.jl")
include("Interface.jl")

using .Attack
using .Model
using .Interface: run, benchmark

export AbstractAttack, WhiteBoxAttack, BlackBoxAttack, name, hyperparameters, craft

export AbstractModel, DifferentiableModel, NonDifferentiableModel, predict, loss, params

export run, benchmark

end
