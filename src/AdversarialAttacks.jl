module AdversarialAttacks

# Include attack interface first
include("attacks/Attack.jl")
using .Attack: AbstractAttack, WhiteBoxAttack, BlackBoxAttack, craft

# Include specific attacks
include("Fgsm.jl")
using .FastGradientSignMethod: FGSM

export FGSM, craft, AbstractAttack, WhiteBoxAttack, BlackBoxAttack

end
