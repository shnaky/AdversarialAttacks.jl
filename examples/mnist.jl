using Random
using Flux
using OneHotArrays
using AdversarialAttacks
using MLDatasets
using Plots

Random.seed!(1234)
println("=== White-Box FGSM Attack Tutorial ===\n")

# ------------------------------------
# 1. Load MNIST subset
# ------------------------------------
train_x, train_y = MLDatasets.MNIST.traindata()        # 28×28×60000, Vector{Int}
train_x = train_x[:, :, 1:6000]                        # use 6000 samples for speed
train_y = train_y[1:6000]

# 4D tensor for CNN: (height, width, channels, batch)
X = Float32.(reshape(train_x, 28, 28, 1, :)) ./ 255     # 28×28×1×N
y = Flux.onehotbatch(train_y, 0:9)                      # 10×N one-hot labels

# ------------------------------------
# 2. Define and train CNN
# ------------------------------------
model = Chain(
  Conv((5, 5), 1 => 6, relu, pad=2), x -> maxpool(x, (2, 2)),  # 28 → 28 → 14
  Conv((5, 5), 6 => 16, relu, pad=0), x -> maxpool(x, (2, 2)), # 14 → 10 → 5
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
dataloader = [(X[:, :, :, i:min(end, i + batch_size - 1)],
  y[:, i:min(end, i + batch_size - 1)])
              for i in 1:batch_size:size(X, 4)]

for epoch in 1:5
  for (x_batch, y_batch) in dataloader
    gs = gradient(m -> loss(m, x_batch, y_batch), model)
    Flux.update!(opt, model, gs[1])
  end
end

flux_model = FluxModel(model)

# Simple accuracy on the training subset (for sanity check)
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

# ------------------------------------
# 3. Pick a demo sample
#    (hand-picked index that works well for FGSM)
# ------------------------------------
demo_idx = 2 # number zero

x0 = X[:, :, :, demo_idx:demo_idx]
label_onehot = y[:, demo_idx]

true_label = argmax(label_onehot)                 # 1–10 index
true_digit = Flux.onecold(label_onehot, 0:9)      # 0–9 digit

sample = (data=x0, label=label_onehot)

# Clean prediction
orig_pred = model(x0)
orig_true_prob = orig_pred[true_label]

clean_label = argmax(orig_pred)[1]
clean_digit = Flux.onecold(orig_pred, 0:9)[1]

println("Chosen sample index: $demo_idx")
println("True digit: $true_digit  (index=$true_label)")
println("Clean prediction: $clean_digit  (index=$clean_label)")
println("Clean probs: ", round.(orig_pred, digits=3))
println("Clean true prob: ", round(orig_true_prob, digits=3))

# ------------------------------------
# 4. Run FGSM white-box attack
# ------------------------------------
ε = 0.0015f0
fgsm_attack = FGSM(epsilon=ε)
println("\nRunning FGSM with ε = $ε ...")

x_adv = craft(sample, flux_model, fgsm_attack)
x_adv = clamp.(x_adv, 0f0, 1f0)  # keep pixels in [0,1]

adv_pred = model(x_adv)
adv_true_prob = adv_pred[true_label]

adv_label = argmax(adv_pred)[1]
adv_digit = Flux.onecold(adv_pred, 0:9)[1]

println("\nOriginal image stats   : min=$(minimum(x0)), max=$(maximum(x0))")
println("Adversarial image stats: min=$(minimum(x_adv)), max=$(maximum(x_adv))")
println("Perturbation L∞ norm   : ", maximum(abs.(x_adv .- x0)))

println("\nAdversarial probs: ", round.(adv_pred, digits=3))
println("True prob: ", round(orig_true_prob, digits=3), " → ",
  round(adv_true_prob, digits=3))

prob_drop_success = adv_true_prob < orig_true_prob
flip_success = (clean_label == true_label) && (adv_label != true_label)

println("[INFO] True-class prob drop success: ",
  prob_drop_success, "  (",
  round(orig_true_prob, digits=3), " → ",
  round(adv_true_prob, digits=3), ")")

println("[INFO] Prediction flip success: ",
  flip_success, "  (clean_digit=", clean_digit,
  ", adv_digit=", adv_digit, ")")

println("Digits summary: true=$true_digit, clean=$clean_digit, adv=$adv_digit")

# ------------------------------------
# 5. Visualization: clean / adv / noise
# ------------------------------------
p1 = heatmap(reshape(x0[:, :, 1, 1], 28, 28),
  title="Original (digit=$true_digit)",
  color=:grays, aspect_ratio=1, size=(300, 300))

p2 = heatmap(reshape(x_adv[:, :, 1, 1], 28, 28),
  title="Adversarial (digit=$adv_digit)",
  color=:grays, aspect_ratio=1, size=(300, 300))

p3 = heatmap(reshape(x_adv[:, :, 1, 1] .- x0[:, :, 1, 1], 28, 28),
  title="Perturbation (ε=$ε)",
  color=:RdBu, aspect_ratio=1, size=(300, 300))

fig = plot(p1, p2, p3, layout=(1, 3), size=(900, 300))
display(fig)

println("\nPress Enter to exit...")
readline()
