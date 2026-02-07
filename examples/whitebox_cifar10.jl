# # White-Box FGSM Attack on CIFAR-10
#
# This tutorial demonstrates how to perform a **white-box adversarial attack**
# using the Fast Gradient Sign Method (FGSM) on a pretrained CNN for CIFAR-10.
#
# **What you will learn:**
# - How to load a pretrained CIFAR-10 model (via Julia artifacts)
# - How to construct an FGSM attack with a custom loss function
# - How to save adversarial examples as images
# - How to evaluate attack success on a single example
#
# ## Prerequisites
#
# Make sure you have the following packages installed:
# `Flux`, `MLDatasets`, `Images`, `Statistics`, and `AdversarialAttacks`.

include("./common/ExperimentUtils.jl")
using .ExperimentUtils

using AdversarialAttacks
using Flux
using MLDatasets
using Images
using ImageCore: colorview
using Statistics
using Printf

println("="^70)
println("White-Box Attack on Flux CNN (CIFAR10)")
println("="^70)

# ## 1. Load and Prepare Data
#
# We load the CIFAR-10 test set and reshape it into 4D tensors (H×W×C×N)
# expected by CNNs. CIFAR-10 has 10 classes: plane, car, bird, cat, deer,
# dog, frog, horse, ship, truck.

function preprocess(dataset)
    x, y = dataset[:]

    x = reshape(x, 32, 32, 3, :)

    ## One-hot encode targets
    y = Flux.onehotbatch(y, 0:9)

    return x, y
end

println("\n[Step 1] Loading CIFAR10 dataset...")

c10_train = CIFAR10(:train)
c10_test = CIFAR10(:test)

x_train, y_train = preprocess(c10_train)
x_test, y_test = preprocess(c10_test)

batchsize = 256
train_loader = Flux.DataLoader(
    (x_train, y_train),
    batchsize = batchsize,
    shuffle = true
);

# ## 2. Load pretrained CNN
#
# We load a pretrained CIFAR-10 model from Julia artifacts. This model is a
# 9-layer CNN that achieves ~85% accuracy on CIFAR-10. The model is loaded
# via LazyArtifacts, which downloads the weights on first use.

println("\n[Step 2] Loading pretrained Flux Model...")

model = ExperimentUtils.load_pretrained_c10_model()

function accuracy(model, x, y)

    ŷ = model(x)

    ŷ_cpu = cpu(ŷ)
    y_cpu = cpu(y)

    ŷ = Flux.onecold(ŷ_cpu)
    y_true = Flux.onecold(y_cpu)

    return mean(ŷ .== y_true)
end

println("Testing model performance...")

acc = accuracy(model, x_test, y_test) * 100

println("  • Clean accuracy: ", round(acc, digits = 2), "%")

# ## 3. Create adversarial test image
#
# We select a single test image (index 453 by default) and run the FGSM attack
# on it. The attack uses a custom loss function (logit cross-entropy) and a
# small perturbation budget (ε=0.02) to generate a subtle adversarial example.
#
# **Note:** Change `image_idx` to create adversarial examples for other images.

println("\n[Step 3] Create Adversarial Test Image...")

image_idx = 453
x_example = Float32.(x_test[:, :, :, image_idx:image_idx])
y_example = y_test[:, image_idx]

loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

fgsm = AdversarialAttacks.FGSM(epsilon = 0.02)

sample = (data = x_example, label = y_example)
println("\n Running attack...")
adv_sample = AdversarialAttacks.attack(fgsm, model, sample, loss = loss_fn)

# Get predictions
original_pred = model(x_example)
adv_pred = model(adv_sample)

# CIFAR-10 class names
class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
original_class = Flux.onecold(original_pred, 1:10)[1]
adv_class = Flux.onecold(adv_pred, 1:10)[1]
true_class = Flux.onecold(y_example, 1:10)
println("\nTrue Class: $(class_names[true_class])")
println("\nOriginal Prediction: $(class_names[original_class])")
println("\nPrediction after AdversarialAttack: $(class_names[adv_class])")

println("\nSaving adversarial example...")
img = adv_sample[:, :, :, 1]
# Ensure values are in [0,1]
img_clamped = clamp.(img, 0.0f0, 1.0f0)

# Convert channel-last → RGB image
img_rgb = colorview(RGB, permutedims(img_clamped, (3, 1, 2)))
Images.save(joinpath(@__DIR__, "cifar10_adversarial.png"), img_rgb)

println("\nDone!")

# ## Common edits to try
#
# - Change `image_idx` to attack different test images (0 to 9999)
# - Adjust `epsilon` to make perturbations stronger (e.g., 0.01 or 0.05)
# - Try different loss functions (e.g., `Flux.crossentropy`)
# - Visualize the perturbation: `img_clamped - x_example[:,:,:,1]`
# - Save the original image for comparison
