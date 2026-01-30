# examples/mlj_whitebox_cnn.jl

"""
White-Box Attack on MLJFlux CNN

Demonstrates FGSM gradient-based attack on a CNN classifier.
Uses evaluate_robustness() to assess attack effectiveness across multiple samples.

# Usage
    julia --project=examples examples/mlj_whitebox_cnn.jl

# With CLI options
    julia --project=examples examples/mlj_whitebox_cnn.jl -n 100 -d mnist

# Options
-n, --num-attack-samples  Number of attack target samples (default: 100)
-d, --dataset             Dataset (mnist/cifar10) (default: mnist)  
-f, --force-retrain       Force model retraining (ignore cache)
"""

include("./common/ExperimentUtils.jl")
using .ExperimentUtils

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

# ==========================================
#   • Experiment: mnist_cnn_whitebox_exp
#   • Clean accuracy: 96.14%
# ==========================================
#   • Experiment: cifar_cnn_whitebox_exp
#   • Clean accuracy: 71.54%
# ==========================================
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

# 1. Load data as images
println("\n[1/4] Loading dataset...")
X_img, y = load_data(config.dataset, config.use_flatten)

# 2. Train MLJFlux ImageClassifier
println("\n[2/4] Training MLJFlux CNN...")
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
println("\n[3/4] Preparing test samples...")

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
