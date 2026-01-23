module AdversarialAttacks

using Random

# External dependencies
using DecisionTree: DecisionTreeClassifier, predict_proba
using Flux: Chain, gradient, OneHotVector, onecold, softmax, crossentropy

include("attacks/Attack.jl")
include("attacks/BasicRandomSearch.jl")
include("attacks/FGSM.jl")
include("Interface.jl")
include("Evaluation.jl")

# Export attack types
export AbstractAttack, WhiteBoxAttack, BlackBoxAttack
export FGSM, BasicRandomSearch, SquareAttack

# Export attack interface functions
export name

# Export model interface functions
export predict, loss, params

# Export evaluation functions
export RobustnessReport
export evaluate_robustness

# Export high-level interface
export attack, benchmark

end
