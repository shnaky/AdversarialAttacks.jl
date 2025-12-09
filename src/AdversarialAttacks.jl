module AdversarialAttacks

# Write your package code here.

include("models/Model.jl")

using .Model

export AbstractModel, DifferentiableModel, NonDifferentiableModel, name, predict, loss, params

end
