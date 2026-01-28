# examples/mlj_mnist_whitebox_cnn.jl

"""
White-Box Attack on MLJFlux CNN

Demonstrates FGSM gradient-based attack on a CNN classifier.
Uses evaluate_robustness() to assess attack effectiveness across multiple samples.
"""

include("Experiments.jl")
using .Experiments
using AdversarialAttacks
using Flux
using CategoricalArrays: levelcode
using ImageCore: channelview

println("="^70)
println("White-Box Attack on MLJFlux CNN (MNIST)")
println("="^70)

config = ExperimentConfig(
    exp_name = "mnist_cnn_whitebox_exp",
    model_file_name = "mnist_forest",
    model_factory = make_mnist_forest,
    dataset = DATASET_MNIST,
    use_flatten = true,
    force_retrain = false,
    split_ratio = 0.8,
    rng = 42,
    model_hyperparams = (n_trees = 200, max_depth = -1)
)

# 1. Load MNIST as images
println("\n[1/4] Loading MNIST dataset...")
# X_img, y = load_mnist_for_mlj()
# H = 28
# W = 28
# C = 1
# N = 1

X_img, y = load_cifar10_for_mlj()
H = 32
W = 32
C = 3
N = 1


# 2. Train MLJFlux ImageClassifier
println("\n[2/4] Training MLJFlux CNN...")

# config = ExperimentConfig("mnist_cnn_whitebox", 0.8, 42)

# mach, meta = get_or_train(
#     make_mnist_cnn,
#     "mnist_cnn_whitebox",
#     config = config,
#     force_retrain = false,
#     epochs = 10,
#     batch_size = 64,
#     use_flatten = false,
# )

config = ExperimentConfig("cifar10_cnn_whitebox", 0.8, 42)

mach, meta = get_or_train(
    make_cifar_cnn,
    "cifar_cnn",
    dataset = :cifar10,
    config = config,
    force_retrain = false,
    epochs = 1,
    batch_size = 128,
    use_flatten = false,
)

accuracy = meta["accuracy"]
test_idx = meta["test_idx"]
y_test = meta["y_test"]

println("  • Experiment: ", config.name)
println("  • Clean accuracy: ", round(accuracy * 100, digits = 2), "%")

# 3. Extract Flux model
flux_model = extract_flux_model(mach)

# 4. Prepare test samples
println("\n[3/4] Preparing test samples...")

N_SAMPLES = 100
test_data = []

for i in 1:min(N_SAMPLES, length(test_idx))
    idx = test_idx[i]
    x_img = X_img[idx]
    true_label_idx = levelcode(y_test[i])

    x_array = Float32.(channelview(x_img))
    x_flux = reshape(x_array, H, W, C, N)

    # Verify correct classification
    pred = flux_model(x_flux)
    pred_label = argmax(pred[:, 1])

    if pred_label == true_label_idx
        y_onehot = Flux.onehot(true_label_idx, 1:10)
        push!(test_data, (data = x_flux, label = y_onehot))
    end
end

println("  • Selected $(length(test_data)) correctly classified samples")

# 5. Run FGSM white-box attack
println("\n[4/4] Running FGSM white-box attack (ε=0.1)...")

fgsm = FGSM(epsilon = 0.1f0)

wb_report = evaluate_robustness(
    flux_model,
    fgsm,
    test_data,
    num_samples = length(test_data)
)

# Print results
println("\n" * "="^70)
println("ROBUSTNESS EVALUATION RESULTS")
println("="^70)
println(wb_report)

println("\n" * "="^70)
println("✓ Evaluation complete!")
println("="^70)
