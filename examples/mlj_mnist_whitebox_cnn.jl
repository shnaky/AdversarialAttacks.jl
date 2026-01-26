# examples/mlj_mnist_whitebox_cnn.jl

"""
White-Box Attack on MLJFlux CNN

Demonstrates FGSM gradient-based attack on a CNN classifier.
Uses evaluate_robustness() to assess attack effectiveness across multiple samples.
"""

include("Experiments.jl")
using .Experiments: load_mnist_for_mlj, make_mnist_cnn, extract_flux_model,
    ExperimentConfig, run_experiment
using AdversarialAttacks
using Flux
using CategoricalArrays: levelcode
using ImageCore: channelview

println("="^70)
println("White-Box Attack on MLJFlux CNN (MNIST)")
println("="^70)

# 1. Load MNIST as images
println("\n[1/4] Loading MNIST dataset...")
X_img, y = load_mnist_for_mlj()

# 2. Train MLJFlux ImageClassifier
println("\n[2/4] Training MLJFlux CNN...")
cnn_model = make_mnist_cnn(epochs = 10, batch_size = 64)
config = ExperimentConfig("mnist_cnn_whitebox", 0.8, 42)
result = run_experiment(cnn_model, X_img, y; config = config)

println("  • Experiment: ", config.name)
println("  • Clean accuracy: ", round(result.report.accuracy * 100, digits = 2), "%")

# 3. Extract Flux model
flux_model = extract_flux_model(result.mach)

# 4. Prepare test samples
println("\n[3/4] Preparing test samples...")

N_SAMPLES = 100
test_data = []

for i in 1:min(N_SAMPLES, length(result.test_idx))
    idx = result.test_idx[i]
    x_img = X_img[idx]
    true_label_idx = levelcode(result.y_test[i])

    x_array = Float32.(channelview(x_img))
    x_flux = reshape(x_array, 28, 28, 1, 1)

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
