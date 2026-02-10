"""
    make_prediction_function(model)

Create a unified prediction function for evaluation only.
This is NOT passed to attack() - only used for getting predictions.

# Arguments
- `model`: Either a Flux-style model or an MLJ Machine.

# Returns
- A function that takes input data and returns prediction probabilities as a vector.

# Note
This function handles different model types (Flux neural networks vs MLJ models)
and input shapes (vectors vs matrices) to provide a consistent prediction interface.
"""
function make_prediction_function(model::Machine)
    return function (x_data)
        if x_data isa AbstractVector
            x_row = reshape(x_data, 1, :)
        else
            x_row = reshape(vec(x_data), 1, :)
        end

        X_tbl = table(x_row)
        pred_dist = predict(model, X_tbl)[1]

        class_levels = levels(pred_dist)
        probs = [pdf(pred_dist, level) for level in class_levels]

        return probs
    end
end

"""
    make_prediction_function(model)

Flux model (Neural Network) version.

# Arguments
- `model`: Flux-compatible neural network model.

# Returns
- A function that takes input data and returns prediction vector.
"""
function make_prediction_function(model::Chain)
    return function (x)
        output = model(x)
        return vec(output)
    end
end

"""
    make_prediction_function(model::DecisionTreeClassifier)
DecisionTree.jl classifier version.

# Arguments
- `model`: A DecisionTreeClassifier from DecisionTree.jl.
# Returns
- A function that takes a flat input vector and returns prediction probabilities.

# Note
This function reshapes the input vector into a 1-row matrix, calls `predict_proba`, and returns the probabilities as a vector.
"""
function make_prediction_function(model::DecisionTreeClassifier)
    return function (x_flat)
        x_row = reshape(Float64.(x_flat), 1, :)
        return predict_proba(model, x_row)
    end
end
