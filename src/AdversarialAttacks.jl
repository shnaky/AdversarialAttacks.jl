module AdversarialAttacks

include("attacks/Attack.jl")
include("models/Model.jl")
include("attacks/BlackBoxAttacks.jl")

using .Attack
using .Model
using .Attack: AbstractAttack, WhiteBoxAttack, BlackBoxAttack, craft
using .BlackBoxAttacks: BasicRandomSearch, SquareAttack

export AbstractAttack, WhiteBoxAttack, BlackBoxAttack, name, hyperparameters, craft
export AbstractModel, DifferentiableModel, NonDifferentiableModel, predict, loss, params
export BasicRandomSearch, SquareAttack

end
