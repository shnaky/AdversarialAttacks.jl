include("Experiments.jl")
using .Experiments: load_mnist_for_mlj, make_mnist_cnn, extract_flux_model
using AdversarialAttacks
using Flux
using CategoricalArrays: levelcode
using ImageCore: channelview

# 1. Load MNIST as images used for the CNN
X_img, y = load_mnist_for_mlj()

# 2. Train MLJFlux ImageClassifier via the common experiment setup
cnn_model = make_mnist_cnn(epochs = 5, batch_size = 64)
config = Experiments.ExperimentConfig("mnist_cnn_mljflux_whitebox", 0.8, 42)

result = Experiments.run_experiment(cnn_model, X_img, y; config = config)

println("Experiment: ", config.name)
println("Clean CNN test accuracy: ", result.report.accuracy)

# 3. Extract the underlying Flux model for white-box access
flux_model = extract_flux_model(result.mach)

# 4. Build a demo sample from one correctly classified test image
x_img = X_img[result.test_idx[1]]   # 28×28 ColorTypes.Gray{Float32}
x_array = Float32.(channelview(x_img))  # 1×28×28 Array{Float32,3}
x = reshape(x_array, 28, 28, 1, 1)      # 28×28×1×1 Array{Float32,4} H×W×C×N

true_label_idx = levelcode(result.y_test[1])     # 1–10
y_onehot = Flux.onehot(true_label_idx, 1:10)

sample = (data = x, label = y_onehot)

# 5. Configure FGSM attack from the library
fgsm = FGSM(; epsilon = 0.0015f0)

# 6. Run the attack via the generic interface
adv_x = attack(fgsm, flux_model, sample)

# 7. Log gradients/perturbation norm/success metrics
x_clean = sample.data
x_adv = adv_x
p_clean = flux_model(x_clean)
p_adv = flux_model(x_adv)

clean_conf = p_clean[true_label_idx, 1]
adv_conf = p_adv[true_label_idx, 1]

linf_norm = maximum(abs.(x_adv .- x_clean))

clean_pred = Flux.onecold(p_clean, 1:10)[1]
adv_pred = Flux.onecold(p_adv, 1:10)[1]

println("\n=== FGSM on MLJFlux CNN (single sample) ===")
println("True label index          : ", true_label_idx)
println(
    "Clean prediction          : ",
    clean_pred,
    "  (p_true = ",
    round(Float64(clean_conf), digits = 3),
    ")",
)
println(
    "Adversarial prediction    : ",
    adv_pred,
    "  (p_true = ",
    round(Float64(adv_conf), digits = 3),
    ")",
)
println("L∞ norm of perturbation   : ", linf_norm)
println(
    "Prediction flip success   : ",
    clean_pred == true_label_idx && adv_pred != true_label_idx,
)
println("True-class prob drop      : ", round(Float64(clean_conf - adv_conf), digits = 3))
