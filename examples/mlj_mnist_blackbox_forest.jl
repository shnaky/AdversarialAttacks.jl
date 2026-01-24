include("Experiments.jl")
using .Experiments
using AdversarialAttacks
using MLJ: mode, fitted_params, table
using CategoricalArrays: levelcode, levels
using Distributions: pdf

# 1. Load MNIST images and flatten for a tabular model
X_img, y = load_mnist_for_mlj()
X_flat = flatten_images(X_img)

# 2. Build and train a RandomForestClassifier as black-box model
forest = make_mnist_forest(rng = 42, n_trees = 200, max_depth = -1)
config = ExperimentConfig("mnist_forest_blackbox", 0.8, 42)
result = run_experiment(forest, X_flat, y; config = config)

println("Experiment: ", config.name)
println("Accuracy on test set (RandomForest): ", result.report.accuracy)

# 3. Sanity check: black-box prediction API on a few test samples
y_test_subset = result.y_test[1:5]
pred_probs = blackbox_predict(result.mach, X_flat[result.test_idx[1:5], :])
pred_labels = mode.(pred_probs)

println("First 5 true labels      : ", y_test_subset)
println("First 5 predicted labels : ", pred_labels)

# 4. Configure a BasicRandomSearch black-box attack on the RandomForest model

# Pick one test sample (flattened feature vector) and its true label index
idx = result.test_idx[1]
x_vec = X_flat[idx, :]                 # 1×d row (tabular feature vector)
x_vec_f = Float32.(collect(x_vec))     # Vector{Float32}
true_label_idx = levelcode(result.y_test[1])

# Build a sample tuple consistent with the BasicRandomSearch interface
sample = (data = x_vec_f, label = true_label_idx)

# Instantiate the BasicRandomSearch attack
# max_iter acts as a query budget proxy
brs = BasicRandomSearch(; epsilon = 0.05, max_iter = 200)

# Run the black-box attack against the MLJ machine
x_adv = attack(brs, result.mach, sample)

# 5. Evaluate clean vs adversarial predictions using only the black-box API

# Helper: query the black-box model and return class probabilities for a single vector
predict_proba_fn = function (x_flat::AbstractVector)
    # Wrap x_flat as a single-row table for MLJ
    x_row = permutedims(x_flat)   # 1×d Matrix
    X_tbl = table(x_row)
    probs = blackbox_predict(result.mach, X_tbl)[1]   # UnivariateFinite
    return collect(pdf.(probs, levels(probs)))        # Vector of probabilities
end

# Clean prediction
probs_clean = predict_proba_fn(sample.data)
clean_prob = probs_clean[true_label_idx]
clean_pred = argmax(probs_clean)

# Adversarial prediction
probs_adv = predict_proba_fn(x_adv)
adv_prob = probs_adv[true_label_idx]
adv_pred = argmax(probs_adv)

# Simple metrics: treat max_iter as query count
query_count = brs.max_iter
conf_drop = clean_prob - adv_prob
flipped = clean_pred != adv_pred

println("\n=== BasicRandomSearch black-box attack on MLJ RandomForest ===")
println("True label index        : ", true_label_idx)
println(
    "Clean prediction        : ",
    clean_pred,
    "  (p_true = ",
    round(Float64(clean_prob), digits = 3),
    ")",
)
println(
    "Adversarial prediction  : ",
    adv_pred,
    "  (p_true = ",
    round(Float64(adv_prob), digits = 3),
    ")",
)
println("Query count             : ", query_count)
println("Prediction flip success : ", flipped)
println("True-class prob drop    : ", round(Float64(conf_drop), digits = 3))
