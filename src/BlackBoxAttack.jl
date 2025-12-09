"""
BlackBoxAttack(parameters)

Subtype of AbstractAttack. Can be used to create an adversarial example in the black-box setting.

'parameters' can be used to pass attack parameters as a dict
"""

struct BlackBoxAttack <: AbstractAttack
    parameters::Dict{String,Any}

    function BlackBoxAttack(parameters::Dict{String,Any}=Dict{String,Any}()) 
        new(parameters)
    end
end

"""
    perform_attack(attack::WhiteBoxAttack, model, sample)

Performs a white-box adversarial attack on the given model using the provided sample.
Returns the adversarial example generated from the sample.

# Arguments
- attack::WhiteBoxAttack: An instance of the WhiteBoxAttack.
- model: The machine learning (deep learning) model to be attacked.
- sample: The input sample to be changed.
"""

function perform_attack(attack::BlackBoxAttack, model, sample)
    return sample
end