module AdversarialAttacks

# External dependencies
using DecisionTree: DecisionTreeClassifier, predict_proba
using Flux: Chain, gradient, OneHotVector, onecold, softmax, crossentropy
using MLJ: predict, levels, Machine, machine, table
using Distributions: pdf
using LinearAlgebra: norm
using Random: seed!, MersenneTwister, default_rng, AbstractRNG, randperm
using Statistics: mean

include("attacks/Attack.jl")
include("attacks/BasicRandomSearch.jl")
include("attacks/FGSM.jl")
include("models/predict_proba.jl")
include("Evaluation.jl")

# Export attack types
export AbstractAttack, WhiteBoxAttack, BlackBoxAttack
export FGSM, BasicRandomSearch

# Export attack interface functions
export name, attack

# Export evaluation functions
export RobustnessReport
export evaluate_robustness, make_prediction_function, evaluation_curve

end
