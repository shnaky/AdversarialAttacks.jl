include("Experiments.jl")
using .Experiments

using MLJ

X_img, y = load_mnist_for_mlj()

X_flat = flatten_images(X_img)

DecisionTreeClassifier = @load DecisionTreeClassifier pkg = DecisionTree

model = DecisionTreeClassifier(
    max_depth = 5,
)

config = ExperimentConfig("mnist_decision_tree", 0.8, 42)
result = run_experiment(model, X_flat, y; config = config)

println("Experiment: ", config.name)
println("Accuracy on test set: ", result.report.accuracy)
