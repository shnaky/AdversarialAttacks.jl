# examples/mlj_mnist_blackbox_tree.jl

"""
Black-Box Attack on Decision Tree Classifier

Demonstrates query-based adversarial attacks on a traditional ML model
using BasicRandomSearch. Unlike neural networks, decision trees have
no gradients, making only black-box attacks feasible.
"""

include("Experiments.jl")
using .Experiments
using AdversarialAttacks
using Flux
using MLJ: mode, predict, table
using CategoricalArrays: levelcode
using Printf

function main()
    println("="^70)
    println("Black-Box Attack on Decision Tree Classifier (MNIST)")
    println("="^70)

    # =========================================================================
    # [Step 1] Load and Prepare Data
    # =========================================================================
    println("\n[Step 1] Loading MNIST dataset...")

    X_img, y = load_mnist_for_mlj()
    X_flat = flatten_images(X_img)

    println("  • Dataset: $(size(X_flat, 1)) samples, $(size(X_flat, 2)) features")

    # =========================================================================
    # [Step 2] Train Decision Tree
    # =========================================================================
    println("\n[Step 2] Training Decision Tree Classifier...")

    tree_model = make_mnist_tree(rng = 42, max_depth = 10)

    config = ExperimentConfig("mnist_tree_blackbox", 0.8, 42)
    result = run_experiment(tree_model, X_flat, y; config = config)

    println("  • Experiment: ", config.name)
    println("  • Clean accuracy: ", round(result.report.accuracy * 100, digits = 2), "%")
    println("  • Tree depth: 10 levels")

    # =========================================================================
    # [Step 3] Prepare Test Samples
    # =========================================================================
    println("\n[Step 3] Preparing test samples...")

    N_SAMPLES = 100
    test_data = []

    for i in 1:min(N_SAMPLES, length(result.test_idx))
        idx = result.test_idx[i]
        x_flat = Float32.(Vector(X_flat[idx, :]))

        true_label = result.y_test[i]
        true_label_idx = levelcode(true_label)

        # Check if correctly classified
        x_row = reshape(x_flat, 1, :)
        X_tbl = table(x_row)
        pred_prob = predict(result.mach, X_tbl)[1]
        pred_label = mode(pred_prob)

        if pred_label == true_label
            y_onehot = Flux.onehot(true_label_idx, 1:10)
            push!(
                test_data, (
                    data = x_flat,
                    label = y_onehot,
                    true_idx = true_label_idx,
                    true_label_cat = true_label,
                )
            )
        end
    end

    println("  • Selected $(length(test_data)) correctly classified samples")

    # =========================================================================
    # [Step 4] Black-Box Attack Evaluation
    # =========================================================================
    println("\n[Step 4] Running Black-Box Attack (BasicRandomSearch with ε=0.1, 200 iter)...")

    attack_config = BasicRandomSearch(epsilon = 0.1f0, max_iter = 200)

    bb_report = evaluate_robustness(
        result.mach,
        attack_config,
        test_data,
        num_samples = length(test_data)
    )

    # =========================================================================
    # [Step 5] Results
    # =========================================================================
    println("\n" * "="^70)
    println("ROBUSTNESS EVALUATION RESULTS")
    println("="^70)

    bb_asr = bb_report.attack_success_rate * 100

    println("\n╔═════════════════════════════╦═══════════════╗")
    println("║ Metric                      ║  Black-Box    ║")
    println("╠═════════════════════════════╬═══════════════╣")
    println("║ Model                       ║  DecisionTree ║")
    println("║ Attack Method               ║  RandomSearch ║")
    @printf("║ Attack Success Rate (ASR)   ║   %5.1f%%      ║\n", bb_asr)
    @printf(
        "║ Successful Attacks          ║   %3d/%3d      ║\n",
        bb_report.num_successful_attacks, bb_report.num_clean_correct
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
    @printf("║ Queries per Sample          ║    200        ║\n")
    println("╚═════════════════════════════╩═══════════════╝")

    # =========================================================================
    # [Step 6] Key Insights
    # =========================================================================
    println("\n" * "="^70)
    println("KEY INSIGHTS")
    println("="^70)
    @printf(
        """
        **Model Characteristics**:
          • Single decision tree (max depth: 10)
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
        Decision trees are vulnerable to black-box attacks despite lacking gradients.
        Random search can find adversarial examples by exploring the feature space
        through iterative querying. Single trees are typically less robust than
        ensemble methods like RandomForest.
        """,
        result.report.accuracy * 100,
        bb_asr,
        bb_report.num_successful_attacks,
        bb_report.num_clean_correct,
        bb_report.linf_norm_mean,
        bb_report.robustness_score * 100,
        bb_report.adv_accuracy * 100
    )

    # =========================================================================
    # [Step 7] Detailed Report
    # =========================================================================
    println("\n" * "="^70)
    println("DETAILED ROBUSTNESS REPORT")
    println("="^70)
    println(bb_report)

    println("\n" * "="^70)
    println("✓ Evaluation complete!")
    println("="^70)

    return nothing
end

main()
