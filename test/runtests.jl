using AdversarialAttacks
using Test

using Flux: Chain, Dense, softmax, onehot
using Random: seed!, MersenneTwister
using DecisionTree: fit! as dt_fit!, predict_proba, load_data
using MLJ: fit! as mlj_fit!, table, machine, @load, @load_iris
using DecisionTree: DecisionTreeClassifier
using CategoricalArrays: CategoricalArray, levelcode, categorical
using MLJDecisionTreeInterface: RandomForestClassifier
using Statistics: mean

include("attacks/Attack.jl")
include("attacks/FGSM.jl")
include("attacks/BasicRandomSearch.jl")
include("models/predict_proba.jl")
include("Evaluation.jl")
