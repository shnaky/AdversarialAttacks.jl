"""
    BasicRandomSearch(parameters::Dict=Dict{String,Any}())

Subtype of BlackBoxAttack. Can be used to create an adversarial example in the black-box setting using random search.

# Arguments
- 'parameters': can be used to pass attack parameters as a dict
"""
struct BasicRandomSearch <: BlackBoxAttack
    parameters::Dict{String,Any}

    function BasicRandomSearch(parameters::Dict{String,Any}=Dict{String,Any}())
        new(parameters)
    end
end


"""
    SquareAttack(parameters::Dict=Dict{String,Any}())

Subtype of BlackBoxAttack. Can be used to create an adversarial example in the black-box setting using the square attack algorithm.

# Arguments
- parameters: can be used to pass attack parameters as a dict
"""
struct SquareAttack <: BlackBoxAttack
    parameters::Dict{String,Any}

    function SquareAttack(parameters::Dict{String,Any}=Dict{String,Any}())
        new(parameters)
    end
end


"""
    craft(sample, model, attack::BasicRandomSearch)

Performs a black-box adversarial attack on the given model using the provided sample using Basic Random Search.

# Arguments
- sample: The input sample to be changed.
- model::AbstractModel: The machine learning (deep learning, classical machine learning) model to be attacked.
- attack::BasicRandomSearch: An instance of the BasicRandomSearch (BlackBox) attack.

# Returns
- Adversarial example (same type and shape as `sample`).
"""
function craft(sample, model::AbstractModel, attack::BasicRandomSearch)
    return sample
end

"""
    craft(sample, model, attack::SquareAttack)

Performs a black-box adversarial attack on the given model using the provided sample using the SquareAttack algorithm.

# Arguments
- sample: The input sample to be changed.
- model::AbstractModel: The machine learning (deep learning, classical machine learning) model to be attacked.
- attack::SquareAttack: An instance of the SquareAttack (BlackBox) attack.

# Returns
- Adversarial example (same type and shape as `sample`).
"""
function craft(sample, model::AbstractModel, attack::SquareAttack)
    return sample
end
