"""
White-Box attack on CIFAR10

Demonstrates white-box adversarial attack on a Flux CNN trained on CIFAR10 and saves one adversarial example as an Image.
"""

include("Experiments.jl")
using .Experiments
using AdversarialAttacks
using Flux
using MLDatasets  
using Images #, FileIO
using Statistics
using Printf
#using Plots


println("="^70)
println("White-Box Attack on Flux CNN (CIFAR10)")
println("="^70)

# =============================================================================
# [Step 1] Load and Prepare Data
# =============================================================================

function preprocess(dataset)
    x, y = dataset[:]

    x = reshape(x, 32, 32, 3, :)

    # One-hot encode targets
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
    batchsize=batchsize, 
    shuffle=true);


# =============================================================================
# [Step 2] Load CNN
# =============================================================================
println("\n[Step 2] Loading pretrained Flux Model...")

model = Experiments.load_pretrained_c10_model() 

function accuracy(model, x, y)

    ŷ = model(x)

    ŷ_cpu = cpu(ŷ)
    y_cpu  = cpu(y)

    ŷ = Flux.onecold(ŷ_cpu)
    y_true = Flux.onecold(y_cpu)

    return mean(ŷ .== y_true)
end

println("Testing model performance...")

acc = accuracy(model, x_test, y_test) * 100

println("  • Clean accuracy: ", round(acc, digits = 2), "%")

# =============================================================================
# [Step 3] Create Adversarial Test Image
# NOTE: change the sample index to create other examples :)
# =============================================================================
println("\n[Step 3] Create Adversarial Test Image...")

image_idx = 453
x_example = Float32.(x_test[:, :, :, image_idx:image_idx])  
y_example = y_test[:, image_idx] 

loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

fgsm = AdversarialAttacks.FGSM(epsilon = 0.02)

sample = (data=x_example, label=y_example)
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
img_clamped = clamp.(img, 0f0, 1f0)

# Convert channel-last → RGB image
img_rgb = colorview(RGB, permutedims(img_clamped, (3, 1, 2)))
Images.save("img_rgb.png", img_rgb)

println("\nDone!")
