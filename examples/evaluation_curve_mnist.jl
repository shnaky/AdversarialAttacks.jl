# examples/evaluation_curve_mnist.jl

include("Experiments.jl")
using .Experiments: load_mnist_for_mlj, extract_flux_model, get_or_train, make_mnist_cnn, ExperimentConfig
using AdversarialAttacks
using Flux
using CategoricalArrays: levelcode
using ImageCore: channelview
using Plots

function run_eval_curve()
    println("="^70)
    println("Evaluation Curve on MNIST Dataset using FGSM Attack")
    println("="^70)

    # ==========================================================================
    # [Step 1] Train/Load CNN Model
    # ==========================================================================
    println("\n[Step 1] Loading/Training MLJFlux CNN on MNIST...")

    mach, meta = get_or_train(
        make_mnist_cnn,
        "evaluation_curve_mnist",
        force_retrain = false,
        epochs = 10,
        batch_size = 64,
        use_flatten = false,
    )

    raw_model = extract_flux_model(mach)

    # Ensure model has softmax at the end
    if raw_model isa Chain && raw_model.layers[end] != softmax
        flux_model = Chain(raw_model, softmax)
        println("  ✓ Added softmax to model")
    else
        flux_model = raw_model
        println("  ✓ Model already has softmax")
    end

    println("  • Clean accuracy: ", round(meta["accuracy"] * 100, digits = 2), "%")

    # ==========================================================================
    # [Step 2] Prepare Test Samples
    # ==========================================================================
    println("\n[Step 2] Preparing test samples...")

    X_img, y = load_mnist_for_mlj()
    N_SAMPLES = 100

    test_data = []

    for i in 1:length(meta["test_idx"])
        if length(test_data) >= N_SAMPLES
            break
        end

        idx = meta["test_idx"][i]
        x_img = X_img[idx]
        true_label_idx = levelcode(meta["y_test"][i])

        # Convert to Flux format (28×28×1×1)
        x_array = Float32.(channelview(x_img))
        x_flux = reshape(x_array, 28, 28, 1, 1)

        # Check if correctly classified
        pred = flux_model(x_flux)
        pred_label = argmax(pred[:, 1])

        if pred_label == true_label_idx
            y_onehot = Flux.onehot(true_label_idx, 1:10)
            push!(test_data, (data = x_flux, label = y_onehot, true_idx = true_label_idx))
        end
    end

    println("  • Selected $(length(test_data)) correctly classified samples")

    # ==========================================================================
    # [Step 3] White-Box Attack (FGSM)
    # ==========================================================================
    println("\n[Step 3] Running White-Box Attack FGSM with different ε...")

    # TODO: list comperhesion
    epsilons = [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0007, 0.0008, 0.0009, 0.001]
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

    # ==========================================================================
    # [Step 4] Plot Metrics
    # ==========================================================================
    println("Plotting Metrics...")
    plot(
        wb_report[:epsilons],
        wb_report[:adv_accuracy],
        label="Adversarial Accuracy",
        linewidth=2,
    )
    plot!(
        wb_report[:epsilons],
        wb_report[:attack_success_rate],
        label="Attack Success Rate",
        linewidth=2,
    )

    xlabel!("ε")
    ylabel!("Score")
    title!("Metrics vs ε")

end

run_eval_curve()
