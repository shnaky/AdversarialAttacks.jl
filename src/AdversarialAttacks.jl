module AdversarialAttacks

include("attacks/Attack.jl")
using .Attack: AbstractAttack, WhiteBoxAttack, BlackBoxAttack, craft

include("Fgsm.jl")
using .FastGradientSignMethod: FGSM

export FGSM, craft, AbstractAttack, WhiteBoxAttack, BlackBoxAttack

end

