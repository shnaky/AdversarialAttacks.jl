using OneHotArrays: OneHotVector
using DecisionTree

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

function _basic_random_search_core(x0, true_label::Int, predict_proba::Function, ε; bounds=nothing)
    # Work in flattened space for coordinate-wise updates
    x_flat = vec(Float32.(x0))
    ndims = length(x_flat)
    perm = randperm(ndims)

    if bounds === nothing
        lb, ub = zeros(eltype(x_flat), ndims), ones(eltype(x_flat), ndims)
    else
        lb = [b[1] for b in bounds]
        ub = [b[2] for b in bounds]
    end

    # Initial probability of the true class
    probs = predict_proba(x_flat)
    last_prob = probs[true_label]

    for i in 1:ndims
        diff = zeros(eltype(x_flat), ndims)
        diff[perm[i]] = ε

        # Left direction
        x_left = clamp.(x_flat .- diff, lb, ub)
        probs_left = predict_proba(x_left)
        left_prob = probs_left[true_label]

        if left_prob < last_prob
            last_prob = left_prob
            x_flat = x_left
        else
            # Right direction
            x_right = clamp.(x_flat .+ diff, lb, ub)
            probs_right = predict_proba(x_right)
            right_prob = probs_right[true_label]

            if right_prob < last_prob
                last_prob = right_prob
                x_flat = x_right
            end
        end
    end

    # Reshape back to original shape of x0
    return reshape(x_flat, size(x0))
end


"""
    craft(sample, model::AbstractModel, attack::BasicRandomSearch)

Performs a black-box adversarial attack on the given model using the provided sample using the Basic Random Search variant SimBA.

# Arguments
- sample: The input sample to be changed.
- model::AbstractModel: The machine learning (deep learning, classical machine learning) model to be attacked.
- attack::BasicRandomSearch: An instance of the BasicRandomSearch (BlackBox) attack.

# Returns
- Adversarial example (same type and shape as `sample.data`).
"""
function craft(sample, model::AbstractModel, attack::BasicRandomSearch)
    x = sample.data
    y = sample.label

    ε = convert(eltype(x), get(attack.parameters, "epsilon", attack.parameters["epsilon"]))
    bounds = get(attack.parameters, "bounds", nothing)

    true_label = isa(y, OneHotVector) ? Flux.onecold(y) : Int(y)

    # Define a closure that matches the shared interface: x_flat → prob vector
    predict_proba = function (x_flat)
        # reshape back to original shape before passing to model
        x_reshaped = reshape(x_flat, size(x))
        probs = model.model(x_reshaped)
        return probs
    end

    return _basic_random_search_core(x, true_label, predict_proba, ε, bounds=bounds)
end

function dt_predict_proba(model::DecisionTreeClassifier, x_flat::AbstractArray)
    x_vec = reshape(Float64.(x_flat), 1, :)
    probs_list = DecisionTree.predict_proba(model, x_vec)
    return probs_list
end

"""
    craft(sample, model::DecisionTreeClassifier, attack::BasicRandomSearch)

Specialized craft function for DecisionTreeClassifier models.

# Arguments
- `sample`: The input sample to be changed.
- `model::DecisionTreeClassifier`: DecisionTree model to be attacked.
- `attack::BasicRandomSearch`: An instance of BasicRandomSearch attack.

# Returns
- Adversarial example (same type and shape as `sample.data`).
"""
function craft(sample, model::DecisionTreeClassifier, attack::BasicRandomSearch)
    x = sample.data
    y = sample.label

    ε = convert(eltype(x), get(attack.parameters, "epsilon", attack.parameters["epsilon"]))
    bounds = get(attack.parameters, "bounds", nothing)

    # Convert one-hot label to integer if needed (1-based)
    true_label = isa(y, OneHotVector) ? Flux.onecold(y) : Int(y)

    # Closure: x_flat → prob vector
    predict_proba = x_flat -> dt_predict_proba(model, x_flat)

    return _basic_random_search_core(x, true_label, predict_proba, ε, bounds=bounds)
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
