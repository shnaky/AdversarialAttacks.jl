module AdversarialAttacks

include("attacks/Attack.jl")
using .Attack

export AbstractAttack, WhiteBoxAttack, BlackBoxAttack, name, hyperparameters, craft

end
