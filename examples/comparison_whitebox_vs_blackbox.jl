# examples/comparison_whitebox_vs_blackbox.jl

"""
White-Box vs Black-Box Attack Comparison on Neural Networks

Demonstrates the efficiency and effectiveness differences between gradient-based
(white-box) and query-based (black-box) adversarial attacks on the same CNN model.

Usage:
    julia --project=examples examples/comparison_whitebox_vs_blackbox.jl
"""

include("./common/ExperimentUtils.jl")
using .ExperimentUtils

function run_comparison()
    println("="^70)
    println("White-Box vs Black-Box Attack Comparison on Neural Network")
    println("="^70)

    # ==========================================================================
    # [Step 1] Train/Load CNN Model
    # ==========================================================================
    println("\n[Step 1] Loading/Training MLJFlux CNN ...")

    dataset = DATASET_MNIST # DATASET_MNIST, DATASET_CIFAR10
    N_SAMPLES = 200

    # ==========================================
    # ✅ comparison_wb_bb_mnist complete: 97.2%
    #   • Clean accuracy: 97.21%
    # ==========================================
    # ✅ comparison_wb_bb_cifar complete: 67.5%
    #   • Clean accuracy: 67.46%
    # ==========================================
    config = ExperimentConfig(
        exp_name = dataset == DATASET_MNIST ? "comparison_wb_bb_mnist_exp" : "comparison_wb_bb_cifar_exp",
        model_file_name = dataset == DATASET_MNIST ? "comparison_wb_bb_mnist" : "comparison_wb_bb_cifar",
        model_factory = dataset == DATASET_MNIST ? make_mnist_cnn : make_cifar_cnn,
        dataset = dataset,
        use_flatten = false,
        force_retrain = false,
        fraction_train = 0.8,
        rng = 42,
        model_hyperparams = (epochs = 10, batch_size = 64)
    )

    mach, meta = get_or_train(config)
    raw_model = extract_flux_model(mach)

    # Ensure model has softmax at the end
    if raw_model isa Chain && raw_model.layers[end] != softmax
        flux_model = Chain(raw_model, softmax)
        println("  ✓ Added softmax to model")
    else
        flux_model = raw_model
        println("  ✓ Model already has softmax")
    end

    accuracy = meta["accuracy"]
    test_idx = meta["test_idx"]
    y_test = meta["y_test"]

    println("  • Clean accuracy: ", round(accuracy * 100, digits = 2), "%")

    # ==========================================================================
    # [Step 2] Prepare Test Samples
    # ==========================================================================
    println("\n[Step 2] Preparing test samples...")

    X_img, _ = load_data(config.dataset, config.use_flatten)
    Xtest_img = X_img[test_idx]
    y_pred_test = predict_mode(mach, Xtest_img)

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
        h, w, c = dataset_shape(Val(dataset))
        x_flux = reshape(x_array, h, w, c, 1)

        true_label_idx = levelcode(true_label)
        y_onehot = onehot(true_label_idx, 1:length(levels(y_test)))

        push!(test_data, (data = x_flux, label = y_onehot, true_idx = true_label_idx))
    end

    println("  • Selected $(length(test_data)) correctly classified samples")

    # ==========================================================================
    # [Step 3] White-Box Attack (FGSM)
    # ==========================================================================
    println("\n[Step 3] Running White-Box Attack (FGSM with ε=0.1)...")

    fgsm = FGSM(epsilon = 0.1f0)

    wb_perturbations = Float64[]
    wb_conf_drops = Float64[]

    wb_time = @elapsed begin
        for (i, sample) in enumerate(test_data)
            pred_clean = flux_model(sample.data)
            clean_conf = pred_clean[sample.true_idx, 1]

            x_adv = attack(fgsm, flux_model, sample)

            pred_adv = flux_model(x_adv)
            adv_conf = pred_adv[sample.true_idx, 1]

            linf = maximum(abs.(x_adv .- sample.data))
            push!(wb_perturbations, Float64(linf))
            push!(wb_conf_drops, Float64(clean_conf - adv_conf))

            if i % 20 == 0
                print("  Progress: $i/$(length(test_data)) samples\r")
            end
        end
    end

    println("  Progress: $(length(test_data))/$(length(test_data)) samples ✓")
    println("  • Completed in $(round(wb_time, digits = 2))s")

    # Evaluate using robustness report
    wb_report = evaluate_robustness(
        flux_model,
        fgsm,
        test_data,
        num_samples = length(test_data),
    )

    # ==========================================================================
    # [Step 4] Black-Box Attack (BasicRandomSearch)
    # ==========================================================================
    println("\n[Step 4] Running Black-Box Attack (BasicRandomSearch with ε=0.2, 100 iter)...")

    brs = BasicRandomSearch(epsilon = 0.2f0, max_iter = 100)

    bb_perturbations = Float64[]
    bb_conf_drops = Float64[]

    bb_time = @elapsed begin
        for (i, sample) in enumerate(test_data)
            pred_clean = flux_model(sample.data)
            clean_conf = pred_clean[sample.true_idx, 1]

            x_adv = attack(brs, flux_model, sample)

            pred_adv = flux_model(x_adv)
            adv_conf = pred_adv[sample.true_idx, 1]

            linf = maximum(abs.(x_adv .- sample.data))
            push!(bb_perturbations, Float64(linf))
            push!(bb_conf_drops, Float64(clean_conf - adv_conf))

            if i % 20 == 0
                print("  Progress: $i/$(length(test_data)) samples\r")
            end
        end
    end

    println("  Progress: $(length(test_data))/$(length(test_data)) samples ✓")
    println("  • Completed in $(round(bb_time, digits = 2))s")

    # Evaluate using robustness report
    bb_report = evaluate_robustness(
        flux_model,
        brs,
        test_data,
        num_samples = length(test_data),
    )

    # ==========================================================================
    # [Step 5] Comparison Results
    # ==========================================================================
    println("\n" * "="^70)
    println("COMPARISON RESULTS")
    println("="^70)

    n_samples = length(test_data)
    bb_asr = bb_report.attack_success_rate * 100
    wb_asr = wb_report.attack_success_rate * 100


    println("\n╔═════════════════════════════╦═══════════════╦═══════════════╗")
    println("║ Metric                      ║  White-Box    ║  Black-Box    ║")
    println("╠═════════════════════════════╬═══════════════╬═══════════════╣")
    println("║ Attack Method               ║  FGSM         ║  RandomSearch ║")
    @printf(
        "║ Attack Success Rate (ASR)   ║   %5.1f%%      ║   %5.1f%%      ║\n",
        wb_report.attack_success_rate * 100,
        bb_report.attack_success_rate * 100,
    )
    @printf(
        "║ Successful Attacks          ║   %3d/%3d      ║   %3d/%3d      ║\n",
        wb_report.num_successful_attacks,
        wb_report.num_clean_correct,
        bb_report.num_successful_attacks,
        bb_report.num_clean_correct,
    )
    println("╠═════════════════════════════╬═══════════════╬═══════════════╣")
    @printf(
        "║ Clean Accuracy              ║   %5.1f%%      ║   %5.1f%%      ║\n",
        wb_report.clean_accuracy * 100,
        bb_report.clean_accuracy * 100,
    )
    @printf(
        "║ Adversarial Accuracy        ║   %5.1f%%      ║   %5.1f%%      ║\n",
        wb_report.adv_accuracy * 100,
        bb_report.adv_accuracy * 100,
    )
    @printf(
        "║ Robustness Score (1-ASR)    ║   %5.1f%%      ║   %5.1f%%      ║\n",
        wb_report.robustness_score * 100,
        bb_report.robustness_score * 100,
    )
    println("╠═════════════════════════════╬═══════════════╬═══════════════╣")
    @printf(
        "║ Avg L∞ Perturbation         ║   %.4f      ║   %.4f      ║\n",
        wb_report.linf_norm_mean,
        bb_report.linf_norm_mean,
    )
    @printf(
        "║ Max L∞ Perturbation         ║   %.4f      ║   %.4f      ║\n",
        wb_report.linf_norm_max,
        bb_report.linf_norm_max,
    )
    println("╠═════════════════════════════╬═══════════════╬═══════════════╣")
    @printf("║ Queries per Sample          ║      1        ║    %3d        ║\n", brs.max_iter)
    @printf(
        "║ Total Time (s)              ║   %6.2f      ║   %6.2f      ║\n",
        wb_time,
        bb_time,
    )
    @printf(
        "║ Time per Sample (ms)        ║   %6.1f      ║   %6.1f      ║\n",
        (wb_time / n_samples) * 1000,
        (bb_time / n_samples) * 1000,
    )
    println("╚═════════════════════════════╩═══════════════╩═══════════════╝")
    # ==========================================================================
    # [Step 6] Key Insights
    # ==========================================================================
    if bb_asr > 0 && wb_asr > 0
        effectiveness_ratio = wb_asr / bb_asr

        println("\n" * "="^70)
        println("KEY INSIGHTS")
        println("="^70)
        @printf(
            """
            **Effectiveness**:
              • White-box is %.1f× more effective (%.1f%% vs %.1f%% ASR)

            **Efficiency**:
              • White-box is 100× more query-efficient (1 vs 100 queries)
              • White-box is %.1f× faster (%.1f ms vs %.1f ms per sample)

            **Perturbation Size**:
              • White-box uses smaller perturbations (%.4f vs %.4f L∞)
              • White-box achieves higher ASR with %.1f%% smaller perturbations

            **Robustness**:
              • Model robustness vs white-box: %.1f%%
              • Model robustness vs black-box: %.1f%%

            **Trade-off Summary**:
            White-box attacks require full model access (gradients) but are significantly
            more effective and efficient. Black-box attacks are more realistic for attacking
            deployed systems where only query access is available.
            """,
            effectiveness_ratio,
            wb_asr,
            bb_asr,
            wb_time / bb_time,
            (wb_time / n_samples) * 1000,
            (bb_time / n_samples) * 1000,
            wb_report.linf_norm_mean,
            bb_report.linf_norm_mean,
            ((bb_report.linf_norm_mean - wb_report.linf_norm_mean) / bb_report.linf_norm_mean) * 100,
            wb_report.robustness_score * 100,
            bb_report.robustness_score * 100
        )
    else
        println("\n⚠️  One or both attacks failed (0% ASR). Try adjusting epsilon or max_iter.")
    end

    # ==========================================================================
    # [Step 7] Detailed Reports
    # ==========================================================================
    println("\n" * "="^70)
    println("DETAILED WHITE-BOX REPORT")
    println("="^70)
    println(wb_report)

    println("\n" * "="^70)
    println("DETAILED BLACK-BOX REPORT")
    println("="^70)
    println(bb_report)

    println("\n" * "="^70)
    println("✓ Comparison complete!")
    println("="^70)

    return nothing
end

# Run the comparison
run_comparison()
