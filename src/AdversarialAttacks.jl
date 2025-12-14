module AdversarialAttacks

# Write your package code here.
include("attacks/Attack.jl")
include("attacks/BlackBoxAttacks.jl")

using .Attack: AbstractAttack, WhiteBoxAttack, BlackBoxAttack, craft
using .BlackBoxAttacks: BasicRandomSearch, SquareAttack

export AbstractAttack, WhiteBoxAttack, BlackBoxAttack, name, hyperparameters, craft, BasicRandomSearch, SquareAttack

end
