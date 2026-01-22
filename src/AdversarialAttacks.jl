module AdversarialAttacks

using Random

include("attacks/Attack.jl")
include("attacks/BlackBoxAttacks.jl")
include("attacks/WhiteBox.jl")
include("Interface.jl")
include("Evaluation.jl")

# Export attack types
export AbstractAttack, WhiteBoxAttack, BlackBoxAttack
export FGSM, BasicRandomSearch, SquareAttack

# Export attack interface functions
export craft, name, hyperparameters

# Export model interface functions
export predict, loss, params

# Export evaluation functions
export RobustnessReport
export evaluate_robustness

# Export high-level interface
export attack, benchmark

end
