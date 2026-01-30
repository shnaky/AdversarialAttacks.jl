module AdversarialAttacks
using Random

# External dependencies
using DecisionTree: DecisionTreeClassifier, predict_proba
using Flux: Chain, gradient, OneHotVector, onecold, softmax, crossentropy
using MLJ: predict, levels, Machine, machine, table
using Distributions: pdf
using LinearAlgebra: norm

include("attacks/Attack.jl")
include("attacks/BasicRandomSearch.jl")
include("attacks/FGSM.jl")
include("Evaluation.jl")

# Export attack types
export AbstractAttack, WhiteBoxAttack, BlackBoxAttack
export FGSM, BasicRandomSearch, SquareAttack

# Export attack interface functions
export name, attack

# Export model interface functions
export predict, loss, params

# Export evaluation functions
export RobustnessReport
export evaluate_robustness, make_prediction_function, evaluation_curve, benchmark

end
