# examples/mlj_mnist_blackbox_forest.jl

"""
Black-Box Attack on MLJ RandomForest

Demonstrates query-based adversarial attack on a traditional ML ensemble model.
Unlike neural networks, tree-based models have no gradients, making only
black-box attacks feasible.
"""

include("Experiments.jl")
using .Experiments
using AdversarialAttacks
using MLJ: mode, predict, table
using CategoricalArrays: levelcode
using Flux  # For onehot encoding
using Printf

println("="^70)
println("Black-Box Attack on RandomForest Classifier (MNIST)")
println("="^70)

dataset = DATASET_MNIST # DATASET_MNIST, DATASET_CIFAR10
N_SAMPLES = 100

# ==========================================
#   • Experiment: mnist_blackbox_forest_exp
#   • Clean accuracy: 96.51%
# ==========================================
#   • Experiment: cifar_blackbox_forest_exp
#   • Clean accuracy: 46.03%
# ==========================================
config = ExperimentConfig(
    exp_name = dataset == DATASET_MNIST ? "mnist_blackbox_forest_exp" : "cifar_blackbox_forest_exp",
    model_file_name = dataset == DATASET_MNIST ? "mnist_blackbox_forest" : "cifar_blackbox_forest",
    model_factory = make_forest,
    dataset = dataset,
    use_flatten = true,
    force_retrain = false,
    split_ratio = 0.8,
    rng = 42,
    model_hyperparams = (n_trees = 200, max_depth = -1)
)

# =============================================================================
# [Step 1] Load and Prepare Data
# =============================================================================
println("\n[Step 1] Loading MNIST dataset...")

X_flat, y = load_data(config.dataset, config.use_flatten) # Flatten images for tabular model

println("  • Dataset: $(size(X_flat, 1)) samples, $(size(X_flat, 2)) features")

# =============================================================================
# [Step 2] Train RandomForest
# =============================================================================
println("\n[Step 2] Training RandomForest Classifier...")

mach, meta = get_or_train(config)

accuracy = meta["accuracy"]
test_idx = meta["test_idx"]
y_test = meta["y_test"]

println("  • Experiment: ", config.exp_name)
println("  • Clean accuracy: ", round(accuracy * 100, digits = 2), "%")

# =============================================================================
# [Step 3] Prepare Test Samples
# =============================================================================
println("\n[Step 3] Preparing test samples...")

n_available = min(N_SAMPLES, length(test_idx))
test_data = []

for i in 1:n_available
    idx = test_idx[i]
    x_vec = Float32.(Vector(X_flat[idx, :]))

    # Get true label
    true_label = y_test[i]
    true_label_idx = levelcode(true_label)

    # Check if correctly classified
    x_row = reshape(x_vec, 1, :)
    X_tbl = table(x_row)
    pred_prob = predict(mach, X_tbl)[1]
    pred_label = mode(pred_prob)

    if pred_label == true_label
        y_onehot = Flux.onehot(true_label_idx, 1:10)
        push!(
            test_data, (
                data = x_vec,
                label = y_onehot,
                true_idx = true_label_idx,
                true_label_cat = true_label,
            )
        )
    end
end

println("  • Selected $(length(test_data)) correctly classified samples")

# =============================================================================
# [Step 4] Black-Box Attack Evaluation
# =============================================================================
println("\n[Step 4] Running Black-Box Attack (BasicRandomSearch with ε=0.1, 200 iter)...")

brs = BasicRandomSearch(epsilon = 0.1f0, max_iter = 200)

bb_report = evaluate_robustness(
    mach,
    brs,
    test_data,
    num_samples = length(test_data)
)

# =============================================================================
# [Step 5] Results
# =============================================================================
println("\n" * "="^70)
println("ROBUSTNESS EVALUATION RESULTS")
println("="^70)

println("\n╔═════════════════════════════╦═══════════════╗")
println("║ Metric                      ║  Black-Box    ║")
println("╠═════════════════════════════╬═══════════════╣")
println("║ Model                       ║  RandomForest ║")
println("║ Attack Method               ║  RandomSearch ║")
@printf(
    "║ Attack Success Rate (ASR)   ║   %5.1f%%      ║\n",
    bb_report.attack_success_rate * 100
)
@printf(
    "║ Successful Attacks          ║   %3d/%3d      ║\n",
    bb_report.num_successful_attacks,
    bb_report.num_clean_correct,
)
println("╠═════════════════════════════╬═══════════════╣")
@printf(
    "║ Clean Accuracy              ║   %5.1f%%      ║\n",
    bb_report.clean_accuracy * 100
)
@printf(
    "║ Adversarial Accuracy        ║   %5.1f%%      ║\n",
    bb_report.adv_accuracy * 100
)
@printf(
    "║ Robustness Score (1-ASR)    ║   %5.1f%%      ║\n",
    bb_report.robustness_score * 100
)
println("╠═════════════════════════════╬═══════════════╣")
@printf(
    "║ Avg L∞ Perturbation         ║   %.4f      ║\n",
    bb_report.linf_norm_mean
)
@printf(
    "║ Max L∞ Perturbation         ║   %.4f      ║\n",
    bb_report.linf_norm_max
)
println("╠═════════════════════════════╬═══════════════╣")
@printf(
    "║ Queries per Sample          ║    %3d        ║\n",
    brs.max_iter
)
println("╚═════════════════════════════╩═══════════════╝")


# =============================================================================
# [Step 6] Key Insights
# =============================================================================
println("\n" * "="^70)
println("KEY INSIGHTS")
println("="^70)
@printf(
    """
    **Model Characteristics**:
      • RandomForest (200 trees, unlimited depth)
      • No gradients available → only black-box attacks feasible
      • Clean accuracy: %.1f%%

    **Attack Performance**:
      • Black-box ASR: %.1f%% (%d/%d successful)
      • Average perturbation: %.4f L∞ norm
      • Queries per sample: 200

    **Robustness**:
      • Model robustness: %.1f%% (1 - ASR)
      • Adversarial accuracy: %.1f%%

    **Conclusion**:
    Tree-based ensembles are vulnerable to black-box attacks despite lacking
    gradients. Random search can find adversarial examples through iterative
    query-based exploration of the decision boundary.
    """,
    accuracy * 100,
    bb_report.attack_success_rate * 100,
    bb_report.num_successful_attacks,
    bb_report.num_clean_correct,
    bb_report.linf_norm_mean,
    bb_report.robustness_score * 100,
    bb_report.adv_accuracy * 100
)

# =============================================================================
# [Step 7] Detailed Report
# =============================================================================
println("\n" * "="^70)
println("DETAILED ROBUSTNESS REPORT")
println("="^70)
println(bb_report)

println("\n" * "="^70)
println("✓ Evaluation complete!")
println("="^70)
