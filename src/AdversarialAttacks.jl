module AdversarialAttacks

using Flux
using MLDatasets

include("attacks/Attack.jl")
include("models/Model.jl")
include("attacks/BlackBoxAttacks.jl")
include("attacks/WhiteBox.jl")
include("models/FluxModels.jl")
include("Interface.jl")

using .Attack
using .Attack: AbstractAttack, WhiteBoxAttack, BlackBoxAttack, craft

using .Model
using .Attack: AbstractAttack, WhiteBoxAttack, BlackBoxAttack, craft
using .BlackBoxAttacks: BasicRandomSearch, SquareAttack
using .FastGradientSignMethod: FGSM
using .FluxModels
using .Interface: attack, benchmark

export AbstractAttack, WhiteBoxAttack, BlackBoxAttack, name, hyperparameters, craft
export AbstractModel, DifferentiableModel, NonDifferentiableModel, predict, loss, params
export BasicRandomSearch, SquareAttack, FGSM
export FluxModel
export attack, benchmark

end