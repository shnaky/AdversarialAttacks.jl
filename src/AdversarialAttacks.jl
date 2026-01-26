module AdversarialAttacks

# External dependencies
using DecisionTree: DecisionTreeClassifier, predict_proba
using Flux: Chain, gradient, OneHotVector, onecold, softmax, crossentropy
using MLJ: predict, levels, Machine, machine, table
using Distributions: pdf

using Random: randperm
using Downloads: download
using BSON: @load
using Artifacts

include("attacks/Attack.jl")
include("attacks/BasicRandomSearch.jl")
include("attacks/FGSM.jl")
include("Interface.jl")
include("Evaluation.jl")

# Export attack types
export AbstractAttack, WhiteBoxAttack, BlackBoxAttack
export FGSM, BasicRandomSearch, SquareAttack
export load_pretrained_c10_model, plot_adversarial_example 

# Export attack interface functions
export name

# Export model interface functions
export predict, loss, params

# Export evaluation functions
export RobustnessReport
export evaluate_robustness, make_prediction_function

# Export high-level interface
export attack, benchmark

end
