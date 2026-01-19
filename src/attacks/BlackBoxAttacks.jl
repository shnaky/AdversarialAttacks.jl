"""
    BasicRandomSearch(parameters::Dict=Dict{String,Any}())

Subtype of BlackBoxAttack. Can be used to create an adversarial example in the black-box setting using random search.

# Arguments
- 'parameters': can be used to pass attack parameters as a dict
"""
struct BasicRandomSearch <: BlackBoxAttack
    parameters::Dict{String,Any}

    function BasicRandomSearch(parameters::Dict=Dict{String,Any}())
        new(Dict{String,Any}(parameters))
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

    function SquareAttack(parameters::Dict=Dict{String,Any}())
        new(parameters)
    end
end


"""
    craft(sample, model, attack::BasicRandomSearch)

Performs a black-box adversarial attack on the given model using the provided sample using the Basic Random Search variant SimBA.

# Arguments
- sample: The input sample to be changed.
- model::AbstractModel: The machine learning (deep learning, classical machine learning) model to be attacked.
- attack::BasicRandomSearch: An instance of the BasicRandomSearch (BlackBox) attack.

# Returns
- Adversarial example (same type and shape as `sample`).
"""
function craft(sample, model::AbstractModel, attack::BasicRandomSearch)
    x = sample.data
    y = sample.label

    ε = convert(eltype(x), get(attack.parameters, "epsilon", attack.parameters["epsilon"]))
    bounds = get(attack.parameters, "bounds", nothing)

    ndims = length(x)
    perm = randperm(ndims)

    if bounds === nothing
        lb, ub = zeros(eltype(x), ndims), ones(eltype(x), ndims)
    else
        lb = [b[1] for b in bounds]
        ub = [b[2] for b in bounds]
    end

    pred = model.model(x)
    last_prob = pred[y]
    #println("Initial probability of true class: ", pred)
    for i in 1:ndims
        diff = zeros(eltype(x), ndims)
        diff[perm[i]] = ε
        δ = reshape(diff, size(x))

        x_left = clamp.(x .- δ, lb, ub)
        left_prob = model.model(x_left)[y]
        if left_prob < last_prob
            last_prob = left_prob
            x = x_left
        else
            x_right = clamp.(x .+ δ, lb, ub)
            right_prob = model.model(x_right)[y]
            #print("Iteration $i: Left prob = $left_prob, Right prob = ")
            if right_prob < last_prob
                last_prob = right_prob
                x = x_right
            end
        end
    end
    return x
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
