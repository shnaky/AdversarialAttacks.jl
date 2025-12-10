module AdversarialAttacks

include("attacks/Attack.jl")
include("models/Model.jl")
include("attacks/WhiteBox.jl")
include("models/FluxModels.jl")
include("Interface.jl")

using .Attack
using .Attack: AbstractAttack, WhiteBoxAttack, BlackBoxAttack, craft

using .Model
using .FastGradientSignMethod: FGSM
using .FluxModels
using .Interface: run, benchmark

export AbstractAttack, WhiteBoxAttack, BlackBoxAttack, name, hyperparameters, craft

export AbstractModel, DifferentiableModel, NonDifferentiableModel, FluxModel, predict, loss, params

export FGSM, craft, AbstractAttack

export run, benchmark

end




