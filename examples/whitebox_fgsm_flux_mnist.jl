# # White-Box FGSM Attack on MNIST (Flux)
#
# This tutorial demonstrates how to perform a **white-box adversarial attack**
# using the Fast Gradient Sign Method (FGSM) against a small CNN trained on MNIST.
#
# **What you will learn:**
# - How to train a simple CNN with Flux on MNIST
# - How to construct an FGSM attack with `AdversarialAttacks.jl`
# - How to evaluate whether the attack succeeded
# - How to visualize original vs adversarial images
#
# ## Prerequisites
#
# Make sure you have the following packages installed:
# `Flux`, `MLDatasets`, `OneHotArrays`, `Plots`, and `AdversarialAttacks`.

using Random
using Flux
using OneHotArrays
using AdversarialAttacks
using MLDatasets
using Plots

Random.seed!(1234)
println("=== White-Box FGSM Attack Tutorial ===\n")

# ## 1. Load MNIST subset
#
# We load a subset of MNIST (6000 samples) and reshape it into the 4D tensor
# format that Flux CNNs expect: `(height, width, channels, batch)`.
# MLDatasets returns pixel values already in the `[0, 1]` range.

train_x, train_y = MLDatasets.MNIST.traindata()        # 28×28×60000, Vector{Int}
train_x = train_x[:, :, 1:6000]                        # use 6000 samples for speed
train_y = train_y[1:6000]

X = Float32.(reshape(train_x, 28, 28, 1, :))     # 28×28×1×N
y = Flux.onehotbatch(train_y, 0:9)                      # 10×N one-hot labels

# ## 2. Define and train a CNN
#
# We define a small LeNet-style CNN with two convolutional layers followed by
# three dense layers, ending with a softmax output. The model is trained for
# 5 epochs using the Adam optimizer with cross-entropy loss.

model = Chain(
    Conv((5, 5), 1 => 6, relu, pad = 2), x -> maxpool(x, (2, 2)),  # 28 → 28 → 14
    Conv((5, 5), 6 => 16, relu, pad = 0), x -> maxpool(x, (2, 2)), # 14 → 10 → 5
    Flux.flatten,                                                 # 16*5*5 = 400
    Dense(400, 120, relu),
    Dense(120, 84, relu),
    Dense(84, 10),
    softmax,
)

loss(m, x, y) = Flux.crossentropy(m(x), y)
opt = Flux.setup(Adam(0.001), model)

println("Training for 5 epochs on mini-batches of size 128...")
batch_size = 128
dataloader = [
    (
            X[:, :, :, i:min(end, i + batch_size - 1)],
            y[:, i:min(end, i + batch_size - 1)],
        )
        for i in 1:batch_size:size(X, 4)
]

for epoch in 1:5
    for (x_batch, y_batch) in dataloader
        gs = gradient(m -> loss(m, x_batch, y_batch), model)
        Flux.update!(opt, model, gs[1])
    end
end

# Check training accuracy as a sanity check:

function eval_acc(model, X_test, y_test)
    correct = 0
    for i in 1:size(X_test, 4)
        pred_probs = model(X_test[:, :, :, i:i])
        pred_label = argmax(pred_probs)[1]
        true_label = argmax(y_test[:, i])
        correct += (pred_label == true_label)
    end
    return correct / size(X_test, 4)
end

println("Train-subset acc: $(round(eval_acc(model, X, y) * 100, digits = 2))%")
println("✓ Trained simple CNN on MNIST subset\n")

# ## 3. Pick a demo sample
#
# We select a single correctly classified sample to attack. The sample must be
# wrapped as a named tuple `(data=x, label=y)` — this is the format that
# `AdversarialAttacks.jl` expects.

demo_idx = 25 # number zero

x0 = X[:, :, :, demo_idx:demo_idx]
label_onehot = y[:, demo_idx]

true_label = argmax(label_onehot)                 # 1–10 index
true_digit = Flux.onecold(label_onehot, 0:9)      # 0–9 digit

sample = (data = x0, label = label_onehot)

## Clean prediction
orig_pred = model(x0)
orig_true_prob = orig_pred[true_label]

clean_label = argmax(orig_pred)[1]
clean_digit = Flux.onecold(orig_pred, 0:9)[1]

println("Chosen sample index: $demo_idx")
println("True digit: $true_digit  (index=$true_label)")
println("Clean prediction: $clean_digit  (index=$clean_label)")
println("Clean probs: ", round.(orig_pred, digits = 3))
println("Clean true prob: ", round(orig_true_prob, digits = 3))

# ## 4. Run the FGSM white-box attack
#
# We construct an `FGSM` attack with a small perturbation budget `ε`.
# The `attack()` function computes the adversarial example by taking one
# gradient step in the direction that maximizes the loss.
#
# After the attack, we clamp pixel values back to `[0, 1]`.

ε = 0.05f0
fgsm_attack = FGSM(epsilon = ε)
println("\nRunning FGSM with ε = $ε ...")

x_adv = attack(fgsm_attack, model, sample)
x_adv = clamp.(x_adv, 0.0f0, 1.0f0)  # keep pixels in [0,1]

adv_pred = model(x_adv)
adv_true_prob = adv_pred[true_label]

adv_label = argmax(adv_pred)[1]
adv_digit = Flux.onecold(adv_pred, 0:9)[1]

println("\nOriginal image stats   : min=$(minimum(x0)), max=$(maximum(x0))")
println("Adversarial image stats: min=$(minimum(x_adv)), max=$(maximum(x_adv))")
println("Perturbation L∞ norm   : ", maximum(abs.(x_adv .- x0)))

# ## 5. Evaluate the attack
#
# We check two success criteria:
# - **Probability drop**: Did the true-class probability decrease?
# - **Prediction flip**: Did the predicted label change from the correct one?

println("\nAdversarial probs: ", round.(adv_pred, digits = 3))
println(
    "True prob: ", round(orig_true_prob, digits = 3), " → ",
    round(adv_true_prob, digits = 3)
)

prob_drop_success = adv_true_prob < orig_true_prob
flip_success = (clean_label == true_label) && (adv_label != true_label)

println(
    "[INFO] True-class prob drop success: ",
    prob_drop_success, "  (",
    round(orig_true_prob, digits = 3), " → ",
    round(adv_true_prob, digits = 3), ")"
)

println(
    "[INFO] Prediction flip success: ",
    flip_success, "  (clean_digit=", clean_digit,
    ", adv_digit=", adv_digit, ")"
)

println("Digits summary: true=$true_digit, clean=$clean_digit, adv=$adv_digit")

# ## 6. Visualization
#
# We plot three heatmaps side by side:
# - **Original**: the clean MNIST image
# - **Adversarial**: the perturbed image after the FGSM attack
# - **Perturbation**: the pixel-wise difference, showing where the attack changed the image

p1 = heatmap(
    reshape(x0[:, :, 1, 1], 28, 28),
    title = "Original (digit=$true_digit)",
    color = :grays, aspect_ratio = 1, size = (300, 300)
)

p2 = heatmap(
    reshape(x_adv[:, :, 1, 1], 28, 28),
    title = "Adversarial (digit=$adv_digit)",
    color = :grays, aspect_ratio = 1, size = (300, 300)
)

p3 = heatmap(
    reshape(x_adv[:, :, 1, 1] .- x0[:, :, 1, 1], 28, 28),
    title = "Perturbation (ε=$ε)",
    color = :RdBu, aspect_ratio = 1, size = (300, 300)
)

fig = plot(p1, p2, p3, layout = (1, 3), size = (900, 300))
savefig(fig, joinpath(@__DIR__, "mnist_fgsm.svg")) #hide

# ![FGSM attack on MNIST](mnist_fgsm.svg)

# ## Common edits to try
#
# - Change `ε` (e.g., `0.05f0 → 0.1f0` or `0.01f0`) to make perturbations stronger or weaker.
# - Change `demo_idx` to attack different digits.
# - Increase training epochs or use more samples for a stronger base classifier.
