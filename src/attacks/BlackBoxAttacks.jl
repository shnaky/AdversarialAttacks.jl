"""
    BasicRandomSearch(; epsilon=0.1, bounds=nothing)

Subtype of BlackBoxAttack. Creates adversarial examples using the SimBA random search algorithm.

# Arguments
- `epsilon`: Step size for perturbations (default: 0.1)
- `bounds`: Optional vector of (lower, upper) tuples specifying per-feature bounds.
            If `nothing`, defaults to [0, 1] for all features (suitable for normalized images).
            For tabular data, provide bounds matching feature ranges, e.g.,
            `[(4.3, 7.9), (2.0, 4.4), ...]` for Iris-like data.
"""
struct BasicRandomSearch{T<:Real, B<:Union{Nothing, Vector{<:Tuple{Real,Real}}}} <: BlackBoxAttack
    epsilon::T
    bounds::B
end

function BasicRandomSearch(; epsilon::Real=0.1, bounds=nothing)
    BasicRandomSearch(epsilon, bounds)
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
    n = length(x)

    ε = convert(eltype(x), attack.epsilon)

    # Set up bounds
    if attack.bounds === nothing
        lb = zeros(eltype(x), n)
        ub = ones(eltype(x), n)
    else
        if length(attack.bounds) != n
            throw(DimensionMismatch("bounds length ($(length(attack.bounds))) must match input dimensions ($n)"))
        end
        lb = eltype(x)[b[1] for b in attack.bounds]
        ub = eltype(x)[b[2] for b in attack.bounds]
    end

    perm = randperm(n)
    pred = model.model(x)
    last_prob = pred[y]

    for i in 1:n
        diff = zeros(eltype(x), n)
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
