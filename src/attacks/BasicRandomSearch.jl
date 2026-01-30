"""
    BasicRandomSearch(; epsilon=0.1, max_iter=50, bounds=nothing, rng=Random.default_rng())

Subtype of `BlackBoxAttack`. Creates adversarial examples using the SimBA random search algorithm. 
Based on Guo, C., Gardner, J., You, Y., Wilson, A. G., & Weinberger, K. (2019, May). [Simple black-box adversarial attacks](https://proceedings.mlr.press/v97/guo19a.html). In International conference on machine learning (pp. 2484-2493). PMLR.


# Arguments
- `epsilon`: Step size for perturbations (default: 0.1).
- `max_iter`: Maximum number of iterations for searching (default: 50).
             Each iteration randomly selects a coordinate to perturb.
- `bounds`: Optional vector of (lower, upper) tuples specifying per-feature bounds.
            If `nothing`, defaults to [0, 1] for all features (suitable for normalized images).
            For tabular data, provide bounds matching feature ranges, e.g.,
            `[(4.3, 7.9), (2.0, 4.4), ...]` for Iris-like data.
- `rng`: Random number generator for reproducibility (default: `Random.default_rng()`).
"""
struct BasicRandomSearch{
        T <: Real,
        B <: Union{Nothing, Vector{<:Tuple{Real, Real}}},
        R <: AbstractRNG,
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
        detailed_result = false,
    )
    queries_used = 0

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
    queries_used += 1  # Query: Initial prediction

    initial_label = argmax(probs)

    last_prob = probs[true_label]
    current_label = initial_label

    # Iterate through permuted coordinates, repeat when max_iter exceeds ndims
    for i in 1:max_iter
        diff = zeros(eltype(x_flat), ndims)
        idx = mod1(i, ndims) # 1-based circular indexing
        diff[perm[idx]] = ε

        # Left direction
        x_left = clamp.(x_flat .- diff, lb, ub)
        probs_left = predict_proba_fn(x_left)
        queries_used += 1  # Query: Left direction test

        left_prob = probs_left[true_label]
        left_label = argmax(probs_left)

        if left_prob < last_prob
            last_prob = left_prob
            x_flat = x_left
            current_label = left_label

            # Early stopping if misclassified
            if left_label != true_label
                break
            end
        else
            # Right direction
            x_right = clamp.(x_flat .+ diff, lb, ub)
            probs_right = predict_proba_fn(x_right)
            queries_used += 1  # Query: Right direction test

            right_prob = probs_right[true_label]
            right_label = argmax(probs_right)

            if right_prob < last_prob
                last_prob = right_prob
                x_flat = x_right
                current_label = right_label

                # Early stopping if misclassified
                if right_label != true_label
                    break
                end
            end
        end
    end

    x_adv = reshape(x_flat, size(x0))        # Reshape back to original shape of x0
    success = (current_label != true_label)

    if detailed_result
        return (
            x_adv = x_adv,
            success = success,
            queries_used = queries_used,
            final_label = current_label,
        )
    else
        return x_adv
    end

end


"""
    attack(atk::BasicRandomSearch, model::Chain, sample; detailed_result)

Perform a Black-box Adversarial Attack on the given model using the provided sample using the Basic Random Search variant SimBA.

# Arguments
- `atk::BasicRandomSearch`: An instance of the BasicRandomSearch (Black-box) attack.
- `model::Chain`: The machine learning (deep learning, classical machine learning) model to be attacked.
- `sample`: Input sample as a named tuple with `data` and `label`.
- `detailed_result::Bool=false`: Return format control.
  - `false` (default): Returns adversarial example only (Array).
  - `true`: Returns NamedTuple with metrics (x_adv, success, queries_used, final_label).

# Returns
- If `detailed_result=false`: Adversarial example (same type as `sample.data`).
- If `detailed_result=true`: NamedTuple containing:
  - `x_adv`: Adversarial example.
  - `success::Bool`: Whether attack succeeded.
  - `queries_used::Int`: Number of model queries.
  - `final_label::Int`: Final predicted class.
  
"""
function attack(atk::BasicRandomSearch, model::Chain, sample; detailed_result = false)
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
        detailed_result,
    )
end

"""
    attack(atk::BasicRandomSearch, model::DecisionTreeClassifier, sample; detailed_result)

Perform a Black-box adversarial attack on a DecisionTreeClassifier using BasicRandomSearch (SimBA).

# Arguments
- `atk::BasicRandomSearch`: Attack instance with `epsilon` and optional `bounds`.
- `model::DecisionTreeClassifier`: DecisionTree.jl classifier to attack.
- `sample`: NamedTuple with `data` and `label` fields.
- `detailed_result::Bool=false`: Return format control.
  - `false` (default): Returns adversarial example only (Array).
  - `true`: Returns NamedTuple with metrics (x_adv, success, queries_used, final_label).

# Returns
- If `detailed_result=false`: Adversarial example (same type as `sample.data`).
- If `detailed_result=true`: NamedTuple containing:
  - `x_adv`: Adversarial example.
  - `success::Bool`: Whether attack succeeded.
  - `queries_used::Int`: Number of model queries.
  - `final_label::Int`: Final predicted class.

"""
function attack(atk::BasicRandomSearch, model::DecisionTreeClassifier, sample; detailed_result = false)
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
        detailed_result,
    )
end

"""
    attack(atk::BasicRandomSearch, mach::Machine, sample)

Black-box Adversarial Attack on an MLJ `Machine` (e.g. a `RandomForestClassifier`)
using BasicRandomSearch (SimBA), via `predict`.

- `atk::BasicRandomSearch`: Attack instance with `epsilon` and `max_iter`.
- `mach::Machine`: Trained MLJ machine with probabilistic predictions.
- `sample`: NamedTuple with `data` (feature vector) and `label` (true class index, 1-based).
- `detailed_result::Bool=false`: Return format control.
  - `false` (default): Returns adversarial example only (Array).
  - `true`: Returns NamedTuple with metrics (x_adv, success, queries_used, final_label).

# Returns
- If `detailed_result=false`: Adversarial example (same type as `sample.data`)
- If `detailed_result=true`: NamedTuple containing:
  - `x_adv`: Adversarial example.
  - `success::Bool`: Whether attack succeeded.
  - `queries_used::Int`: Number of model queries.
  - `final_label::Int`: Final predicted class.

"""
function attack(atk::BasicRandomSearch, mach::Machine, sample; detailed_result = false)
    x = sample.data
    y = sample.label
    ε = convert(eltype(x), atk.epsilon)

    # Convert one-hot label to integer if needed (1-based)
    true_label = isa(y, OneHotVector) ? onecold(y) : Int(y)

    predict_proba_fn = function (x_flat)
        # Treat x_flat as a single-row table for MLJ
        x_row = permutedims(x_flat)
        X_tbl = table(x_row)
        probs = predict(mach, X_tbl)[1]
        return collect(pdf.(probs, levels(probs)))
    end

    return _basic_random_search_core(
        x,
        true_label,
        predict_proba_fn,
        ε,
        atk.max_iter,
        atk.rng;
        bounds = atk.bounds,
        detailed_result,
    )
end
