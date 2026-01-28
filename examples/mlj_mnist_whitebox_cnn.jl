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
println("White-Box Attack on MLJFlux CNN")
println("="^70)

dataset = DATASET_MNIST # DATASET_MNIST, DATASET_CIFAR10

# ==========================================
#   • Experiment: mnist_cnn_whitebox_exp
#   • Clean accuracy: 96.14%
# ==========================================
#   • Experiment: cifar_cnn_whitebox_exp
#   • Clean accuracy: 65.5%
# ==========================================
config = ExperimentConfig(
    exp_name = dataset == DATASET_MNIST ? "mnist_cnn_whitebox_exp" : "cifar_cnn_whitebox_exp",
    model_file_name = dataset == DATASET_MNIST ? "mnist_cnn_whitebox" : "cifar_cnn_whitebox",
    model_factory = dataset == DATASET_MNIST ? make_mnist_cnn : make_cifar_cnn,
    dataset = dataset,
    use_flatten = false,
    force_retrain = false,
    split_ratio = 0.8,
    rng = 42,
    model_hyperparams = (epochs = 5, batch_size = 64)
)

# 1. Load data as images
println("\n[1/4] Loading dataset...")
if config.dataset == DATASET_MNIST
    X_img, y = load_mnist_for_mlj()
elseif config.dataset == DATASET_CIFAR10
    X_img, y = load_cifar10_for_mlj()
else
    throw(ArgumentError("Unsupported DatasetType: $config.dataset"))
end
# 2. Train MLJFlux ImageClassifier
println("\n[2/4] Training MLJFlux CNN...")
mach, meta = get_or_train(config)

accuracy = meta["accuracy"]
test_idx = meta["test_idx"]
y_test = meta["y_test"]

println("  • Experiment: ", config.exp_name)
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
    h, w, c = dataset_shapes[config.dataset]
    x_flux = reshape(x_array, h, w, c, 1)

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

println("\n" * "="^70)
println("ROBUSTNESS EVALUATION RESULTS")
println("="^70)
println(wb_report)

println("\n" * "="^70)
println("✓ Evaluation complete!")
println("="^70)
