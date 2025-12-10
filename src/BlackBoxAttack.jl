"""
    BasicRandomSearch(parameters)

Subtype of BlackBoxAttack. Can be used to create an adversarial example in the black-box setting using random search.

'parameters' can be used to pass attack parameters as a dict
"""

struct BasicRandomSearch <: BlackBoxAttack
    parameters::Dict{String,Any}

    function BasicRandomSearch(parameters::Dict{String,Any}=Dict{String,Any}()) 
        new(parameters)
    end
end

"""
    craft(attack::BasicRandomSearch, model, sample)

Performs a black-box adversarial attack on the given model using the provided sample using Basic Random Search.
Returns the adversarial example generated from the sample.

# Arguments
- attack::BasicRandomSearch: An instance of the BasicRandomSearch (BlackBox) attack.
- model: The machine learning (deep learning, classical machine learning) model to be attacked.
- sample: The input sample to be changed.
"""

function craft(attack::BasicRandomSearch, model, sample)
    return sample
end

"""
    SquareAttack(parameters)

Subtype of BlackBoxAttack. Can be used to create an adversarial example in the black-box setting using the square attack algorithm.

'parameters' can be used to pass attack parameters as a dict
"""

struct SquareAttack <: BlackBoxAttack
    parameters::Dict{String,Any}

    function SquareAttack(parameters::Dict{String,Any}=Dict{String,Any}()) 
        new(parameters)
    end
end

"""
    craft(attack::SquareAttack, model, sample)

Performs a black-box adversarial attack on the given model using the provided sample using the SquareAttack algorithm.
Returns the adversarial example generated from the sample.

# Arguments
- attack::SquareAttack: An instance of the SquareAttack (BlackBox) attack.
- model: The machine learning (deep learning, classical machine learning) model to be attacked.
- sample: The input sample to be changed.
"""

function craft(attack::SquareAttack, model, sample)
    return sample
end
