module FastGradientSignMethod

include("attacks/Attack.jl")
using .Attack: WhiteBoxAttack


"""
    FGSM(parameters::Dict{String,Any}=Dict{String,Any}())

A struct that can be used to create a white-box adversarial attack of type Fast Gradient Sign Method.
Subtypes of `WhiteBoxAttack`.

# Arguments
- `parameters::Dict{String,Any}`: A dictionary of parameters for the attack. Defaults to an empty dictionary.
"""
struct FGSM <: WhiteBoxAttack
    parameters::Dict{String,Any}

    function FGSM(parameters::Dict{String,Any}=Dict{String,Any}()) 
        new(parameters)
    end
end

"""
    perform_attack(attack::FGSM, model, sample)

Performs a Fast Gradient Sign Method (FGSM) white-box adversarial attack on the given `model` using the provided `sample`.
Returns the adversarial example generated from the `sample`.

# Arguments
- `attack::FGSM`: An instance of the `FGSM`.
- `model`: The machine learning (deep learning) model to be attacked.
- `sample`: The input sample to be changed.
"""
function craft(attack::FGSM, model, sample)
    return sample
end

export FGSM, craft

end