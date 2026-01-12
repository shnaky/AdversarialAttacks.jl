module FastGradientSignMethod

using ..Attack: WhiteBoxAttack
using ..Model
using ..Model: DifferentiableModel
import ..Attack: craft, hyperparameters
import Flux


"""
    FGSM(parameters::Dict=Dict{String,Any}())

A struct that can be used to create a white-box adversarial attack of type Fast Gradient Sign Method.
Subtypes of `WhiteBoxAttack`.

# Arguments
- `parameters::Dict`: A dictionary of parameters for the attack. Defaults to an empty dictionary.
"""
struct FGSM <: WhiteBoxAttack
    parameters::Dict{String,Any}

    function FGSM(parameters::Dict=Dict{String,Any}())
        new(Dict{String,Any}(parameters))
    end
end

""" 
    hyperparameters(atk::FGSM) -> Dict{String,Any}

Return hyperparameters for an FGSM attack. 
"""
hyperparameters(atk::FGSM) = atk.parameters

"""
    craft(sample, model, attack::FGSM)

Performs a Fast Gradient Sign Method (FGSM) white-box adversarial attack on the given `model` using the provided `sample`.
Returns the adversarial example generated from the `sample`.

# Arguments
- `sample`: The input sample to be changed: tuple (data, label).
- `model`: The machine learning (deep learning) model to be attacked.
- `attack::FGSM`: An instance of the `FGSM`.
"""
function craft(sample, model, attack::FGSM)
    x = sample.data
    y = sample.label
    ε = get(attack.parameters, "epsilon", 0.01)
    grads = Flux.gradient(x_var -> Model.loss(model, x_var, y), x)[1]
    perturbation = ε * sign.(grads)
    adversarial_example = sample.data .+ perturbation
    return adversarial_example
end

export FGSM, craft

end

