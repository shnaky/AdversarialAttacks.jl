# examples/comparison_all_models_attacks.jl

"""
Comprehensive adversarial attack comparison across all integrated models.

Uses ExperimentConfig for unified configuration.

Usage:
    julia --project=examples examples/comparison_all_models_attacks.jl
"""

include("./common/ExperimentUtils.jl")
using .ExperimentUtils

# =========================
# Experiment Configurations
# =========================

exp_name = "comparison_all"
dataset = DATASET_MNIST # DATASET_MNIST, DATASET_CIFAR10
num_of_samples = 10

attackConfigs_FGSM = [
    (FGSM(epsilon = 0.1f0), num_of_samples),
    (FGSM(epsilon = 0.3f0), num_of_samples),
]
attackConfigs_BSR = [
    (BasicRandomSearch(epsilon = 0.1f0, max_iter = 50), num_of_samples),
    (BasicRandomSearch(epsilon = 0.3f0, max_iter = 50), num_of_samples),
]

# ==========================================
# 📦 Loaded cached comparison_mnist_cnn (Acc: 96.1%)
# 📦 Loaded cached comparison_mnist_tree (Acc: 86.5%)
# 📦 Loaded cached comparison_mnist_forest (Acc: 96.2%)
# 📦 Loaded cached comparison_mnist_knn (Acc: 96.6%)
# 📦 Loaded cached comparison_mnist_xgboost (Acc: 97.0%)
# 📦 Loaded cached comparison_mnist_logistic (Acc: 54.2%)
# ==========================================
# 📦 Loaded cached comparison_cifar_cnn (Acc: 65.4%)
# 📦 Loaded cached comparison_cifar_tree (Acc: 28.4%)
# 📦 Loaded cached comparison_cifar_forest (Acc: 42.8%)
# 📦 Loaded cached comparison_cifar_knn (Acc: 33.1%)
# 📦 Loaded cached comparison_cifar_xgboost (Acc: 50.5%)
# 📦 Loaded cached comparison_cifar_logistic (Acc: 41.0%)
# ==========================================

const ALL_CONFIGS = [
    (
        ExperimentConfig(
            exp_name = exp_name,
            model_file_name = dataset == DATASET_MNIST ? "comparison_mnist_cnn" : "comparison_cifar_cnn", #     is_cnn = (exp_config.model_file_name == "comparison_mnist_cnn" || exp_config.model_file_name == "comparison_cifar_cnn")
            model_factory = dataset == DATASET_MNIST ? make_mnist_cnn : make_cifar_cnn,
            dataset = dataset,
            use_flatten = false,
            force_retrain = false,
            fraction_train = 0.8,
            rng = 42,
            model_hyperparams = (epochs = 5,),
        ),
        attackConfigs_FGSM,
    ),
    (
        ExperimentConfig(
            exp_name = exp_name,
            model_file_name = dataset == DATASET_MNIST ? "comparison_mnist_tree" : "comparison_cifar_tree",
            model_factory = make_tree,
            dataset = dataset,
            use_flatten = true,
            force_retrain = false,
            fraction_train = 0.8,
            rng = 42,
            model_hyperparams = (max_depth = 10,)
        ),
        attackConfigs_BSR,
    ),
    (
        ExperimentConfig(
            exp_name = exp_name,
            model_file_name = dataset == DATASET_MNIST ? "comparison_mnist_forest" : "comparison_cifar_forest",
            model_factory = make_forest,
            dataset = dataset,
            use_flatten = true,
            force_retrain = false,
            fraction_train = 0.8,
            rng = 42,
            model_hyperparams = (n_trees = 50,)
        ),
        attackConfigs_BSR,
    ),
    (
        ExperimentConfig(
            exp_name = exp_name,
            model_file_name = dataset == DATASET_MNIST ? "comparison_mnist_knn" : "comparison_cifar_knn",
            model_factory = make_knn,
            dataset = dataset,
            use_flatten = true,
            force_retrain = false,
            fraction_train = 0.8,
            rng = 42,
            model_hyperparams = (K = 10,)
        ),
        attackConfigs_BSR,
    ),
    (
        ExperimentConfig(
            exp_name = exp_name,
            model_file_name = dataset == DATASET_MNIST ? "comparison_mnist_xgboost" : "comparison_cifar_xgboost",
            model_factory = make_xgboost,
            dataset = dataset,
            use_flatten = true,
            force_retrain = false,
            fraction_train = 0.8,
            rng = 42,
            model_hyperparams = (num_round = 50, max_depth = 6)
        ),
        attackConfigs_BSR,
    ),
    (

        ExperimentConfig(
            exp_name = exp_name,
            model_file_name = dataset == DATASET_MNIST ? "comparison_mnist_logistic" : "comparison_cifar_logistic",
            model_factory = make_logistic,
            dataset = dataset,
            use_flatten = true,
            force_retrain = false,
            fraction_train = 0.8,
            rng = 42,
            model_hyperparams = NamedTuple()  # default
        ),
        attackConfigs_BSR,
    ),
]

# =========================
# Helper Functions
# =========================
"""
    prepare_test_samples(mach, meta, n_samples::Int, use_flatten::Bool, is_cnn::Bool)

Prepare test samples in format required by evaluate_robustness.
Returns vector of (data, label) tuples with integer labels.
"""
function prepare_test_samples(mach, meta, n_samples::Int, use_flatten::Bool, is_cnn::Bool, dataset::DatasetType)
    flux_model = is_cnn ? extract_flux_model(mach) : nothing
    test_idx = meta["test_idx"]
    y_test = meta["y_test"]

    X_img, y_full = load_data(dataset, use_flatten)

    label_levels = levels(y_test)  # CategoricalArray level order

    n_available = min(n_samples, length(test_idx))
    test_data = []

    for i in 1:n_available
        global_idx = test_idx[i]
        true_label = y_test[i]

        true_label_obj = y_full[global_idx]
        true_label_idx = levelcode(true_label_obj)

        if is_cnn
            x_img = X_img[global_idx]

            y_mlj = predict_mode(mach, [x_img])[1]
            if y_mlj != true_label
                continue
            end

            # For CNN: reshape to 4D array (28×28×1×1) or (32x32x3xN)
            x_array = Float32.(channelview(x_img))
            h, w, c = dataset_shapes[dataset]
            x_flux = reshape(x_array, h, w, c, 1)

            true_label_idx = levelcode(true_label)
            y_onehot = onehot(true_label_idx, 1:length(label_levels))

            push!(test_data, (data = x_flux, label = y_onehot))
        else
            # For tree-based models: flatten to vector
            x_flat = Float32.(Vector(X_img[global_idx, :]))

            x_row = reshape(x_flat, 1, :)
            X_tbl = table(x_row)
            pred_prob = predict(mach, X_tbl)[1]
            pred_label = mode(pred_prob)

            if pred_label == true_label_obj
                y_onehot = onehot(true_label_idx, 1:length(label_levels))
                push!(test_data, (data = x_flat, label = y_onehot, true_idx = true_label_idx))
            end
        end
    end

    println("  • Selected $(length(test_data)) correctly classified samples")

    return test_data
end

"""
    get_model_for_evaluation(mach, is_cnn::Bool)

Extract the appropriate model for evaluate_robustness.
For CNN, extract Flux model. For others, use MLJ machine.
"""
function get_model_for_evaluation(mach, is_cnn::Bool)
    if is_cnn
        return extract_flux_model(mach)
    else
        return mach
    end
end

"""
    extract_metrics_from_report(report)

Extract metrics from RobustnessReport, handling different field names.
"""
function extract_metrics_from_report(report)
    # Check available fields
    fields = fieldnames(typeof(report))

    # Extract perturbation metrics (handle different possible field names)
    linf_mean = if :mean_perturbation in fields
        report.mean_perturbation
    elseif :mean_linf_norm in fields
        report.mean_linf_norm
    elseif :avg_perturbation in fields
        report.avg_perturbation
    else
        NaN
    end

    linf_max = if :max_perturbation in fields
        report.max_perturbation
    elseif :max_linf_norm in fields
        report.max_linf_norm
    else
        NaN
    end

    return (
        clean_acc = report.clean_accuracy,
        asr = report.attack_success_rate,
        robustness = report.robustness_score,
        linf_mean = report.linf_norm_mean,
        linf_max = report.linf_norm_max,
    )
end

# =========================
# Main Evaluation
# =========================

"""
    evaluate_single_model(model_name::String, factory, use_flatten::Bool, 
                          attack_configs)

Evaluate single model with all its attack configurations.
"""
function evaluate_single_model(exp_config::ExperimentConfig, attack_configs)
    println("\n" * "="^70)
    println("📦 Model: $(exp_config.model_file_name)")
    println("="^70)

    mach, meta = get_or_train(exp_config)
    println("✓ Loaded (Clean Acc: $(round(meta["accuracy"] * 100, digits = 1))%)")

    # Get model in correct format
    is_cnn = (exp_config.model_file_name == "comparison_mnist_cnn" || exp_config.model_file_name == "comparison_cifar_cnn")
    model = get_model_for_evaluation(mach, is_cnn)

    # Run all attacks
    results = []
    for (attack, n_samples) in attack_configs
        attack_name = "$(typeof(attack).name.name)_eps$(attack.epsilon)"

        println("\n⚔️  Attack: $attack_name, Samples: $n_samples")

        try
            # Prepare test data with correct format for model type
            test_samples = prepare_test_samples(mach, meta, n_samples, exp_config.use_flatten, is_cnn, exp_config.dataset)

            println("length(test_samples)", length(test_samples))
            # Run evaluation using built-in function
            report = evaluate_robustness(
                model, attack, test_samples,
                num_samples = length(test_samples)
            )
            println(report)

            # Extract metrics safely
            metrics = extract_metrics_from_report(report)

            # Create result tuple
            result = (
                model = exp_config.model_file_name,
                attack = attack_name,
                clean_acc = metrics.clean_acc,
                asr = metrics.asr,
                robustness = metrics.robustness,
                linf_mean = metrics.linf_mean,
                linf_max = metrics.linf_max,
            )

            push!(results, result)

            # Print concise summary
            println(
                "   ✓ ASR: $(round(result.asr * 100, digits = 1))% | " *
                    "Robust: $(round(result.robustness * 100, digits = 1))%"
            )

        catch e
            println("   ❌ Error: $e")
            # Print stack trace for debugging
            println(stacktrace(catch_backtrace()))
        end
    end

    return results
end

"""
    run_full_comparison()

Run comprehensive comparison across all models and attacks.
"""
function run_full_comparison()
    println("\n" * "🚀"^35)
    println("ADVERSARIAL ROBUSTNESS COMPARISON")
    println("🚀"^35)
    println("Started: $(format(now(), "HH:MM:SS"))")

    all_results = []

    for (exp_config, attack_configs) in ALL_CONFIGS
        results = evaluate_single_model(exp_config, attack_configs)
        append!(all_results, results)
    end

    # Generate summary
    generate_summary_report(all_results)

    return all_results
end

"""
    generate_summary_report(results)

Generate concise summary table.
"""
function generate_summary_report(results)
    if isempty(results)
        println("\n⚠️  No results to display")
        return
    end

    println("\n\n" * "="^80)
    println("📊 SUMMARY")
    println("="^80)

    # Group by attack epsilon
    attack_types = unique([r.attack for r in results])

    for attack_type in attack_types
        println("\n🎯 $attack_type")
        println("-"^80)
        @printf("%-20s %12s %12s %12s\n", "Model", "Clean Acc", "ASR", "Robustness")
        println("-"^80)

        attack_results = filter(r -> r.attack == attack_type, results)
        sort!(attack_results, by = r -> r.robustness, rev = true)

        for r in attack_results
            @printf(
                "%-20s %11.1f%% %11.1f%% %11.1f%%\n",
                r.model, r.clean_acc * 100, r.asr * 100, r.robustness * 100
            )
        end

        # Highlight best
        if !isempty(attack_results)
            best = attack_results[1]
            println("\n   🏆 Most Robust: $(best.model) ($(round(best.robustness * 100, digits = 1))%)")
        end
    end

    println("\n" * "="^80)
    println("Completed: $(format(now(), "HH:MM:SS"))")
    return println("Total Evaluations: $(length(results))")
end

# =========================
# Main Entry Point
# =========================
run_full_comparison()
