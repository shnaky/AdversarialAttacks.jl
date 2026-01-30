# examples/evaluation_curve_mnist.jl

"""
Plotting metrics on White-Box Attack on MLJFlux CNN

Demonstrates an FGSM gradient-based attack on a CNN classifier.
Uses `evaluation_curve()` to evaluate robustness across multiple
perturbation magnitudes (ε values).

Usage:
    julia --project=examples examples/evaluation_curve_mnist.jl
"""

include("./common/ExperimentUtils.jl")
using .ExperimentUtils
using AdversarialAttacks
using Flux
using CategoricalArrays: levelcode
using ImageCore: channelview
using MLJ
using Plots

println("="^70)
println("White-Box Attack on MLJFlux CNN")
println("="^70)

dataset = DATASET_MNIST # DATASET_MNIST, DATASET_CIFAR10
N_SAMPLES = 100

config = ExperimentConfig(
    exp_name = dataset == DATASET_MNIST ? "mnist_cnn_whitebox_exp" : "cifar_cnn_whitebox_exp",
    model_file_name = dataset == DATASET_MNIST ? "mnist_cnn_whitebox" : "cifar_cnn_whitebox",
    model_factory = dataset == DATASET_MNIST ? make_mnist_cnn : make_cifar_cnn,
    dataset = dataset,
    use_flatten = false,
    force_retrain = false,
    fraction_train = 0.8,
    rng = 42,
    model_hyperparams = (epochs = 5, batch_size = 64),
)

# 1. Load data as images
println("\n[1/5] Loading dataset...")
X_img, y = load_data(config.dataset, config.use_flatten)

# 2. Train MLJFlux ImageClassifier
println("\n[2/5] Training MLJFlux CNN...")
mach, meta = get_or_train(config)

acc_meta = meta["accuracy"]
test_idx = meta["test_idx"]
y_test = meta["y_test"]

println("  • Experiment: ", config.exp_name)
println("  • Clean accuracy: ", round(acc_meta * 100, digits = 2), "%")

# 2.5 Recompute MLJ test predictions (optional but consistent)
Xtest_img = X_img[test_idx]
y_pred_test = predict_mode(mach, Xtest_img)

# 3. Extract Flux model
flux_model = extract_flux_model(mach)

# 4. Prepare test samples
println("\n[3/5] Preparing test samples...")

label_levels = levels(y_test)
n_available = min(N_SAMPLES, length(test_idx))
test_data = []

for i in 1:n_available
    idx = test_idx[i]
    x_img = X_img[idx]
    true_label = y_test[i]

    # Use precomputed MLJ prediction for this test index
    y_mlj = y_pred_test[i]
    if y_mlj != true_label
        continue
    end

    x_array = Float32.(channelview(x_img))
    h, w, c = dataset_shapes[config.dataset]
    x_flux = reshape(x_array, h, w, c, 1)

    true_label_idx = levelcode(true_label)
    y_onehot = Flux.onehot(true_label_idx, 1:length(label_levels))

    push!(test_data, (data = x_flux, label = y_onehot))
end

println("  • Selected $(length(test_data)) correctly classified samples")

# 5. Run FGSM white-box attack
println("\n[4/5] Running FGSM white-box attack with different ε...")

epsilons = [i * 1e-4 for i in 0:10]
println("ε = ", epsilons, "\n")

fgsm = FGSM

# Evaluate using robustness report
wb_report = evaluation_curve(
    flux_model,
    fgsm,
    epsilons,
    test_data,
    num_samples = length(test_data),
)

# 5. Plot metrics
println("\n[5/5] Plot metrics...")
eps = wb_report[:epsilons]

p_metrics = plot(
    eps,
    wb_report[:adv_accuracy],
    label="Adversarial Accuracy",
    title="Metrics vs ε",
    ylabel="Score",
)

plot!( p_metrics, eps, wb_report[:attack_success_rate], label="Attack Success Rate",)

p1 = plot(eps, wb_report[:linf_norm_mean], label="L∞ mean", title="Mean norms")
plot!(p1, eps, wb_report[:l2_norm_mean], label="L2 mean")
plot!(p1, eps, wb_report[:l1_norm_mean], label="L1 mean")

p2 = plot(eps, wb_report[:linf_norm_max], label="L∞ max", title="Max norms")
plot!(p2, eps, wb_report[:l2_norm_max], label="L2 max")
plot!(p2, eps, wb_report[:l1_norm_max], label="L1 max")

plot(
    p_metrics,
    p1,
    p2,
    layout=(3,1),
    xlabel="ε",
    legendfontsize=8,
    legend=:outerright,
    linewidth=2,
)
