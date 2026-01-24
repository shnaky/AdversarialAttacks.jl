include("Experiments.jl")
using .Experiments
using MLJ

# 1. Load MNIST as images
X_img, y = load_mnist_for_mlj()

# 2. Build CNN ImageClassifier using MLJFlux
model = make_mnist_cnn(epochs = 5, batch_size = 64)

config = ExperimentConfig("mnist_cnn_mljflux", 0.8, 42)

result = run_experiment(model, X_img, y; config = config)

println("Experiment: ", config.name)
println("Accuracy on test set (CNN): ", result.report.accuracy)
