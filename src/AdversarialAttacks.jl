module AdversarialAttacks

include("Fgsm.jl")
using .FastGradientSignMethod: FGSM, craft

export FGSM, craft

end
