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
        if x_data isa AbstractVector
            x_row = reshape(x_data, 1, :)
        else
            x_row = reshape(vec(x_data), 1, :)
        end

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
