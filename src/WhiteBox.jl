""""
    WhiteBoxAttack(parameters::Dict{String,Any}=Dict{String,Any}())

A struct that can be used to create a white-box adversarial attack.
Subtypes of `AbstractAttack`.

# Arguments
- `parameters::Dict{String,Any}`: A dictionary of parameters for the attack. Defaults to an empty dictionary.
"""
struct WhiteBoxAttack <: AbstractAttack
    parameters::Dict{String,Any}

    function WhiteBoxAttack(parameters::Dict{String,Any}=Dict{String,Any}()) 
        new(parameters)
    end
end

"""
    perform_attack(attack::WhiteBoxAttack, model, sample)

Performs a white-box adversarial attack on the given `model` using the provided `sample`.
Returns the adversarial example generated from the `sample`.

# Arguments
- `attack::WhiteBoxAttack`: An instance of the `WhiteBoxAttack`.
- `model`: The machine learning (deep learning) model to be attacked.
- `sample`: The input sample to be changed.
"""
function perform_attack(attack::WhiteBoxAttack, model, sample)
    return sample
end