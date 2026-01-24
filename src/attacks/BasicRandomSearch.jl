"""
    BasicRandomSearch(; epsilon=0.1, max_iter=50, bounds=nothing, rng=Random.default_rng())

Subtype of BlackBoxAttack. Creates adversarial examples using the SimBA random search algorithm.

# Arguments
- `epsilon`: Step size for perturbations (default: 0.1)
- `max_iter`: Maximum number of iterations for searching (default: 50).
             Each iteration randomly selects a coordinate to perturb.
- `bounds`: Optional vector of (lower, upper) tuples specifying per-feature bounds.
            If `nothing`, defaults to [0, 1] for all features (suitable for normalized images).
            For tabular data, provide bounds matching feature ranges, e.g.,
            `[(4.3, 7.9), (2.0, 4.4), ...]` for Iris-like data.
- `rng`: Random number generator for reproducibility (default: `Random.default_rng()`)
"""
struct BasicRandomSearch{
    T<:Real,
    B<:Union{Nothing,Vector{<:Tuple{Real,Real}}},
    R<:AbstractRNG,
} <: BlackBoxAttack
    epsilon::T
    max_iter::Int
    bounds::B
    rng::R
end

function BasicRandomSearch(;
    epsilon::Real = 0.1,
    max_iter::Int = 50,
    bounds = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    return BasicRandomSearch(epsilon, max_iter, bounds, rng)
end

function _basic_random_search_core(
    x0,
    true_label::Int,
    predict_proba_fn::Function,
    ε,
    max_iter::Int,
    rng::AbstractRNG;
    bounds = nothing,
)
    # Work in flattened space for coordinate-wise updates
    x_flat = vec(Float32.(x0))
    ndims = length(x_flat)
    perm = randperm(rng, ndims)

    if bounds === nothing
        lb, ub = zeros(eltype(x_flat), ndims), ones(eltype(x_flat), ndims)
    else
        if length(bounds) != ndims
            throw(
                DimensionMismatch(
                    "bounds length ($(length(bounds))) must match input dimensions ($ndims)",
                ),
            )
        end
        lb = eltype(x_flat)[b[1] for b in bounds]
        ub = eltype(x_flat)[b[2] for b in bounds]
    end

    # Initial probability of the true class
    probs = predict_proba_fn(x_flat)
    last_prob = probs[true_label]

    # Iterate through permuted coordinates, repeat when max_iter exceeds ndims
    idx = 1
    for _ in 1:max_iter
        diff = zeros(eltype(x_flat), ndims)
        diff[perm[idx]] = ε
        # Update index for next iteration
        idx = idx % ndims + 1

        # Left direction
        x_left = clamp.(x_flat .- diff, lb, ub)
        probs_left = predict_proba_fn(x_left)
        left_prob = probs_left[true_label]

        if left_prob < last_prob
            last_prob = left_prob
            x_flat = x_left
            # Early stopping if misclassified
            if argmax(probs_left) != true_label
                break
            end
        else
            # Right direction
            x_right = clamp.(x_flat .+ diff, lb, ub)
            probs_right = predict_proba_fn(x_right)
            right_prob = probs_right[true_label]

            if right_prob < last_prob
                last_prob = right_prob
                x_flat = x_right
                # Early stopping if misclassified
                if argmax(probs_right) != true_label
                    break
                end
            end
        end
    end

    # Reshape back to original shape of x0
    return reshape(x_flat, size(x0))
end


"""
    attack(atk::BasicRandomSearch, model::Chain, sample)

Perform a black-box adversarial attack on the given model using the provided sample using the Basic Random Search variant SimBA.

# Arguments
- atk::BasicRandomSearch: An instance of the BasicRandomSearch (BlackBox) attack.
- model::Chain: The machine learning (deep learning, classical machine learning) model to be attacked.
- sample: The input sample to be changed.

# Returns
- Adversarial example (same type and shape as `sample.data`).
"""
function attack(atk::BasicRandomSearch, model::Chain, sample)
    x = sample.data
    y = sample.label

    ε = convert(eltype(x), atk.epsilon)

    true_label = isa(y, OneHotVector) ? onecold(y) : Int(y)

    # Define a closure that matches the shared interface: x_flat → prob vector
    predict_proba_fn = function (x_flat)
        # reshape back to original shape before passing to model
        x_reshaped = reshape(x_flat, size(x))
        probs = model(x_reshaped)
        return probs
    end

    return _basic_random_search_core(
        x,
        true_label,
        predict_proba_fn,
        ε,
        atk.max_iter,
        atk.rng;
        bounds = atk.bounds,
    )
end

"""
    attack(atk::BasicRandomSearch, model::DecisionTreeClassifier, sample)

Perform a black-box adversarial attack on a DecisionTreeClassifier using BasicRandomSearch (SimBA).

# Arguments
- `atk::BasicRandomSearch`: Attack instance with `epsilon` and optional `bounds`.
- `model::DecisionTreeClassifier`: DecisionTree.jl classifier to attack.
- `sample`: NamedTuple with `data` and `label` fields.

# Returns
- Adversarial example (same shape as `sample.data`).
"""
function attack(atk::BasicRandomSearch, model::DecisionTreeClassifier, sample)
    x = sample.data
    y = sample.label

    ε = convert(eltype(x), atk.epsilon)

    # Convert one-hot label to integer if needed (1-based)
    true_label = isa(y, OneHotVector) ? onecold(y) : Int(y)

    # Closure: x_flat → prob vector
    predict_proba_fn = function (x_flat)
        x_row = reshape(Float64.(x_flat), 1, :)
        return predict_proba(model, x_row)
    end

    return _basic_random_search_core(
        x,
        true_label,
        predict_proba_fn,
        ε,
        atk.max_iter,
        atk.rng;
        bounds = atk.bounds,
    )
end
