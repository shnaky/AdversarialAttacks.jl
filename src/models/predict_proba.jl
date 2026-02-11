"""
    make_prediction_function(model::Machine)
MLJ machine version.

# Arguments
- `model::Machine`: The machine learning model for which to create a prediction function.

# Returns
- A function that takes input data and returns prediction probabilities.

# Note
This function reshapes the input data to match the expected format for MLJ models and returns the predicted probabilities for each class.
"""

function make_prediction_function(mach::Machine)
    return function (x_data)
        x_vec = x_data isa AbstractVector ? x_data : vec(x_data)

        # x_row should be 1×N, do not use permutedims() which creates N x 1
        x_row = reshape(x_vec, 1, :)

        X_tbl = table(x_row)
        pred_dist = predict(mach, X_tbl)[1]

        class_levels = levels(pred_dist)
        probs = [pdf(pred_dist, level) for level in class_levels]

        return probs
    end
end

"""
    make_prediction_function(model::Chain)
Flux.jl model version.

# Arguments
- `model`: A Flux.jl Chain model.

# Returns
- A function that takes input data and returns prediction probabilities.

# Note
This function assumes that the model's output is already in the form of probabilities (e.g., via a softmax layer).
    If not, additional processing may be needed to convert raw outputs to probabilities.
"""

# used in evaluation function for Flux models
function make_prediction_function(model::Chain)
    return function (x)
        output = model(x)
        return vec(output)
    end
end

# used in attack function for Flux models
function make_prediction_function(model::Chain, x_template)
    x_shape = size(x_template)

    return function (x_flat)
        @assert length(x_flat) == prod(x_shape)
        x_reshaped = reshape(x_flat, x_shape)
        probs = model(x_reshaped)
        return probs
    end
end

"""
    make_prediction_function(model::DecisionTreeClassifier)
DecisionTree.jl classifier version.

# Arguments
- `model`: A DecisionTreeClassifier model from DecisionTree.jl.

# Returns
- A function that takes input data and returns prediction probabilities.

# Note
This function reshapes the input data to match the expected format for DecisionTree.jl and returns the predicted probabilities for each class.
"""

function make_prediction_function(model::DecisionTreeClassifier)
    return function (x_flat)
        x_row = reshape(Float64.(x_flat), 1, :)
        return predict_proba(model, x_row)
    end
end
