module AdversarialAttacks

include("attacks/Attack.jl")
using .Attack: AbstractAttack, WhiteBoxAttack, BlackBoxAttack, craft

include("attacks/WhiteBox.jl")
using .FastGradientSignMethod: FGSM

export FGSM, craft, AbstractAttack, WhiteBoxAttack, BlackBoxAttack

end

