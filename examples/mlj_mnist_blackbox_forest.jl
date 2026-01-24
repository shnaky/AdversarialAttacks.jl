include("Experiments.jl")
using .Experiments
using MLJ

# 1. Load MNIST images and flatten for tabular model
X_img, y = load_mnist_for_mlj()
X_flat = flatten_images(X_img)

# 2. Build a RandomForestClassifier as black-box model
forest = make_mnist_forest(rng = 42, n_trees = 200, max_depth = -1)

config = ExperimentConfig("mnist_forest_blackbox", 0.8, 42)

result = run_experiment(forest, X_flat, y; config = config)

println("Experiment: ", config.name)
println("Accuracy on test set (RandomForest): ", result.report.accuracy)

# 3. Demonstrate pure prediction API for later attacks
#    (here just a tiny sanity check on a few samples)
X_test_subset = result.y_test[1:5]  # labels
pred_probs = blackbox_predict(result.mach, X_flat[result.test_idx[1:5], :])
pred_labels = mode.(pred_probs)

println("First 5 true labels   : ", X_test_subset)
println("First 5 predicted labels: ", pred_labels)
