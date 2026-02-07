using AdversarialAttacks
using Test

using Flux: Chain, Dense, softmax, onehot
using Random: seed!, MersenneTwister
using DecisionTree: fit! as dt_fit!, predict_proba
using MLJ: fit! as mlj_fit!, table, machine, @load
using DecisionTree: DecisionTreeClassifier
using CategoricalArrays: CategoricalArray, levelcode, categorical
using MLJDecisionTreeInterface: RandomForestClassifier

include("attacks/Attack.jl")
include("attacks/FGSM.jl")
include("attacks/BasicRandomSearch.jl")
include("Evaluation.jl")
