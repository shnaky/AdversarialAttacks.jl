# examples/evaluation_curve_mnist.jl

"""

Evaluation Curve Generation for MNIST FGSM Attacks

Demonstrates robustness evaluation curves across different attack strengths.
Uses evaluation_curve() to generate different metrics vs. epsilon curves showing
how model performance degrades under increasing perturbation budgets.

This example creates evaluation curves for white-box attack FGSM
on MNIST digit classification, plotting accuracy degradation as attack strength
increases.

# Usage
    julia --project=examples examples/evaluation_curve.mnist.jl
"""

include("./common/ExperimentUtils.jl")
using .ExperimentUtils
using Plots

println("="^70)
println("White-Box Attack on MLJFlux CNN")
println("="^70)

args = parse_common_args()
arg_num_attack_samples = args["num-attack-samples"]
arg_dataset = dataset_from_string(args["dataset"])
arg_forceretrain = args["force-retrain"]

NUM_ATTACK_SAMPLES = arg_num_attack_samples # default: 100
dataset = arg_dataset # default: DATASET_MNIST , list: DATASET_MNIST, DATASET_CIFAR10
force_retrain = arg_forceretrain # default: false

config = ExperimentConfig(
    exp_name = dataset == DATASET_MNIST ? "mnist_cnn_whitebox_exp" : "cifar_cnn_whitebox_exp",
    model_file_name = dataset == DATASET_MNIST ? "mnist_cnn_whitebox" : "cifar_cnn_whitebox",
    model_factory = dataset == DATASET_MNIST ? make_mnist_cnn : make_cifar_cnn,
    dataset = dataset,
    use_flatten = false,
    force_retrain = force_retrain,
    fraction_train = 0.8,
    rng = 42,
    model_hyperparams = (epochs = 5, batch_size = 64),
)

# ## 1. Load the MNIST dataset
#
# We load the classic **MNIST** handwritten digit dataset using the
# `load_data` helper function from `common/ExperimentUtils.jl`.
#
# The behavior of the loader (e.g. flattening images) is controlled
# via the experiment configuration.
println("\n[1/5] Loading dataset...")
X_img, y = load_data(config.dataset, config.use_flatten)

# ## 2. Train MLJFlux Image Classifier
#
# This step trains our MLJFlux Convolutional Neural Network (CNN) model.
# If a trained model already exists, it will be loaded instead.
# We also print some key metadata about the model, and the trained model
# will be used for further evaluation in later steps.
println("\n[2/5] Training MLJFlux CNN...")
mach, meta = get_or_train(config)

acc_meta = meta["accuracy"]
test_idx = meta["test_idx"]
y_test = meta["y_test"]

println("  • Experiment: ", config.exp_name)
println("  • Clean accuracy: ", round(acc_meta * 100, digits = 2), "%")

# Recompute MLJ test predictions (optional but consistent)
Xtest_img = X_img[test_idx]
y_pred_test = predict_mode(mach, Xtest_img)

# Extract Flux model
flux_model = extract_flux_model(mach)

# ## 3. Prepare Test Samples for the `FGSM` Adversarial Attack
#
# In this step, we select a subset of test samples from the MNIST dataset
# to use for the `FGSM` (Fast Gradient Sign Method) adversarial attack.
# Only correctly classified samples are kept to ensure the attack is valid.
println("\n[3/5] Preparing test samples...")

label_levels = levels(y_test)
n_available = min(NUM_ATTACK_SAMPLES, length(test_idx))
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
    h, w, c = dataset_shape(Val(dataset))
    x_flux = reshape(x_array, h, w, c, 1)

    true_label_idx = levelcode(true_label)
    y_onehot = onehot(true_label_idx, 1:length(label_levels))

    push!(test_data, (data = x_flux, label = y_onehot))
end

println("  • Selected $(length(test_data)) correctly classified samples")

# ## 4. Run the `FGSM` White-Box Attack over Multiple ε Values
#
# In this step, we evaluate the robustness of the trained model against the
# Fast Gradient Sign Method (`FGSM`) white-box attack.
#
# We first construct a vector of perturbation magnitudes ε (epsilon),
# controlling the strength of the adversarial perturbation:
#
#   ε = [0.0, 0.0001, 0.0002, 0.0003, 0.0004,
#        0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
#
# For each ε value, adversarial examples are generated and evaluated using
# the `evaluation_curve` function from `Evaluation.jl`.
# The result is a dictionary containing robustness metrics (e.g. accuracy,
# attack success rate, perturbation norms) as a function of ε.
println("\n[4/5] Running FGSM white-box attack with different ε...")

epsilons = [i * 1.0e-4 for i in 0:10]
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

# ## 5. Plot Adversarial Metrics over ε
#
# In this final step, we visualize the results returned by the `evaluation_curve`
# function. Plotting the metrics against the perturbation strengths ε allows us
# to understand how the model's performance degrades and how the attack
# effectiveness grows as the perturbation increases.
#
# We create three plots:
# 1. **Model Performance Metrics**: adversarial accuracy and attack success rate.
# 2. **Mean Perturbation Norms**: L1, L2, and L∞ norms averaged over samples.
# 3. **Maximum Perturbation Norms**: maximum L1, L2, and L∞ norms observed.
println("\n[5/5] Plot metrics...")
eps = wb_report[:epsilons]

p_metrics = plot(
    eps,
    wb_report[:adv_accuracy],
    label = "Adversarial Accuracy",
    title = "Metrics vs ε",
    ylabel = "Score",
)

plot!(p_metrics, eps, wb_report[:attack_success_rate], label = "Attack Success Rate")

p1 = plot(eps, wb_report[:linf_norm_mean], label = "L∞ mean", title = "Mean norms")
plot!(p1, eps, wb_report[:l2_norm_mean], label = "L2 mean")
plot!(p1, eps, wb_report[:l1_norm_mean], label = "L1 mean")

p2 = plot(eps, wb_report[:linf_norm_max], label = "L∞ max", title = "Max norms")
plot!(p2, eps, wb_report[:l2_norm_max], label = "L2 max")
plot!(p2, eps, wb_report[:l1_norm_max], label = "L1 max")

plot(
    p_metrics,
    p1,
    p2,
    layout = (3, 1),
    xlabel = "ε",
    legendfontsize = 8,
    legend = :outerright,
    linewidth = 2,
)
