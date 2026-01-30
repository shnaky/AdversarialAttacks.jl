# # Black-Box Basic Random Search Attack on Iris (DecisionTree)
#
# This tutorial demonstrates how to perform a **black-box adversarial attack**
# using Basic Random Search (a SimBA-style algorithm) against a Decision Tree
# classifier trained on the Iris dataset.
#
# A **black-box** attack has no access to model gradients or internal
# parameters — it can only *query* the model for predictions. Basic Random
# Search is a greedy coordinate-cycling algorithm: it iterates over features in
# a random order and, for each feature, tries a small perturbation of +ε or −ε.
# If the perturbation reduces the true-class probability, it keeps the change;
# otherwise it reverts it. The search stops early when the predicted class
# flips.
#
# **What you will learn:**
# - How to train a DecisionTree classifier on Iris
# - How to construct a `BasicRandomSearch` attack with `AdversarialAttacks.jl`
# - How to evaluate whether the attack succeeded
# - How to visualize original vs adversarial samples in feature space
#
# ## Prerequisites
#
# Make sure you have the following packages installed:
# `RDatasets`, `DecisionTree`, `Flux`, `OneHotArrays`, `Plots`, and `AdversarialAttacks`.

using Random
using RDatasets
using DecisionTree
using Flux
using OneHotArrays: OneHotVector
using AdversarialAttacks
using Plots

Random.seed!(1234)

# ## 1. Train a DecisionTree on Iris
#
# We load the classic Iris dataset, extract features and labels, and fit a
# `DecisionTreeClassifier` with a maximum depth of 3.

iris = dataset("datasets", "iris")
X = Matrix{Float64}(iris[:, 1:4])
y_str = String.(iris.Species)
classes = ["setosa", "versicolor", "virginica"]

dt_model = DecisionTreeClassifier(
    max_depth = 3,
    min_samples_leaf = 1,
    min_samples_split = 2,
    classes = classes,
)
fit!(dt_model, X, y_str)

println("Trained DecisionTreeClassifier on Iris.")
println("Classes = ", dt_model.classes)

## helper
function predict_class_index(model::DecisionTreeClassifier, x::AbstractVector)
    x_mat = reshape(Float64.(x), 1, :)
    probs = vec(DecisionTree.predict_proba(model, x_mat))
    return argmax(probs)
end

# ## 2. Pick a correctly classified demo sample
#
# We search for a correctly classified versicolor sample to use as our attack
# target. Versicolor sits near the decision boundary with virginica, making it
# a good candidate for a successful perturbation.

demo_idx = findfirst(==("versicolor"), y_str)

for i in 1:size(X, 1)
    y_str[i] == "versicolor" || continue
    xi = X[i, :]
    true_idx_i = findfirst(==(y_str[i]), classes)
    pred_idx_i = predict_class_index(dt_model, xi)
    if pred_idx_i == true_idx_i
        global demo_idx = i
        break
    end
end

x0 = X[demo_idx, :]
label_str = y_str[demo_idx]
true_idx = findfirst(==(label_str), classes)

println("\nChosen demo sample index: ", demo_idx)
println("Feature vector: ", x0)
println("True label string: ", label_str, " (index ", true_idx, ")")

# ## 3. Build the sample NamedTuple
#
# The attack interface expects a named tuple `(data=..., label=...)` where
# `label` is a one-hot encoded vector.

y0 = Flux.onehot(true_idx, 1:length(classes))
sample = (data = Float32.(x0), label = y0)

x0_mat = reshape(Float64.(x0), 1, :)
orig_probs_vec = vec(DecisionTree.predict_proba(dt_model, x0_mat))
orig_true_prob = orig_probs_vec[true_idx]

println("\nOriginal probabilities: ", orig_probs_vec)
println("Original predicted class index = ", argmax(orig_probs_vec))

# ## 4. Run BasicRandomSearch
#
# We configure the attack with:
# - `epsilon = 0.3`: maximum perturbation per feature
# - `bounds`: valid ranges for each Iris feature
# - `max_iter = 100`: number of random search iterations
#
# The algorithm cycles through features in a random order and tries ±ε for
# each one. It greedily keeps any perturbation that reduces the true-class
# probability and stops early if the predicted class flips.

ε = 0.3f0
atk = BasicRandomSearch(
    epsilon = ε,
    bounds = [(4.3f0, 7.9f0), (2.0f0, 4.4f0), (1.0f0, 6.9f0), (0.1f0, 2.5f0)],
    max_iter = 100,
)
println("\nRunning BasicRandomSearch with epsilon = ", ε, " and max_iter = ", atk.max_iter, " ...")
Random.seed!(42)

x_adv = attack(atk, dt_model, sample)

x_adv_mat = reshape(Float64.(x_adv), 1, :)
adv_probs_vec = vec(DecisionTree.predict_proba(dt_model, x_adv_mat))
adv_true_prob = adv_probs_vec[true_idx]

println("\nOriginal feature vector:     ", sample.data)
println("Adversarial feature vector: ", x_adv)

println("\nOriginal probs:     ", orig_probs_vec)
println("Adversarial probs: ", adv_probs_vec)

# ## 5. Evaluate the attack
#
# We check whether the attack decreased the true-class confidence.

println("\nTrue-class probability before attack: ", orig_true_prob)
println("True-class probability after attack:  ", adv_true_prob)

if adv_true_prob < orig_true_prob
    println("\n[INFO] Attack decreased the true-class confidence (success).")
else
    println("\n[INFO] True-class confidence did not decrease.")
end

# ## 6. Visualization
#
# We create two scatter plots showing 2D projections of the Iris dataset
# (features 1 & 2, and features 3 & 4). The original sample is shown as a
# black star and the adversarial sample as an orange star, with an arrow
# indicating the perturbation direction.
#
# **How to read the plots:** The background points (circles, triangles,
# squares) show the training data coloured by class. The black star marks the
# original sample's position and the orange star marks where the adversarial
# perturbation moved it. A gray arrow connects the two. If the attack
# succeeded, the orange star will sit in (or near) a region dominated by a
# different class, meaning the classifier's prediction flipped. If both stars
# overlap, the attack did not find a perturbation that changed the prediction.

idx_setosa = findall(==("setosa"), y_str)
idx_versicolor = findall(==("versicolor"), y_str)
idx_virginica = findall(==("virginica"), y_str)

orig_pred_class = classes[argmax(orig_probs_vec)]
adv_pred_class = classes[argmax(adv_probs_vec)]

## Plot 1: features 1 & 2
p12 = plot(
    xlabel = "SepalLength",
    ylabel = "SepalWidth",
    title = "Iris (features 1&2)",
)
scatter!(
    p12, X[idx_setosa, 1], X[idx_setosa, 2],
    color = :blue, markershape = :circle, alpha = 0.6, label = "setosa",
)
scatter!(
    p12, X[idx_versicolor, 1], X[idx_versicolor, 2],
    color = :green, markershape = :utriangle, alpha = 0.6, label = "versicolor",
)
scatter!(
    p12, X[idx_virginica, 1], X[idx_virginica, 2],
    color = :red, markershape = :square, alpha = 0.6, label = "virginica",
)
scatter!(
    p12, [x0[1]], [x0[2]],
    markersize = 12, color = :black, markershape = :star5,
    markerstrokecolor = :white, label = "original ($orig_pred_class)",
)
scatter!(
    p12, [x_adv[1]], [x_adv[2]],
    markersize = 12, color = :orange, markershape = :star5,
    markerstrokecolor = :white, label = "adversarial ($adv_pred_class)",
)
plot!(
    p12, [x0[1], x_adv[1]], [x0[2], x_adv[2]],
    arrow = true, color = :gray, linewidth = 1.5, label = "",
)

## Plot 2: features 3 & 4
p34 = plot(xlabel = "PetalLength", ylabel = "PetalWidth", title = "Iris (features 3&4)")
scatter!(
    p34, X[idx_setosa, 3], X[idx_setosa, 4],
    color = :blue, markershape = :circle, alpha = 0.6, label = "setosa",
)
scatter!(
    p34, X[idx_versicolor, 3], X[idx_versicolor, 4],
    color = :green, markershape = :utriangle, alpha = 0.6, label = "versicolor",
)
scatter!(
    p34, X[idx_virginica, 3], X[idx_virginica, 4],
    color = :red, markershape = :square, alpha = 0.6, label = "virginica",
)
scatter!(
    p34, [x0[3]], [x0[4]],
    markersize = 12, color = :black, markershape = :star5,
    markerstrokecolor = :white, label = "original ($orig_pred_class)",
)
scatter!(
    p34, [x_adv[3]], [x_adv[4]],
    markersize = 12, color = :orange, markershape = :star5,
    markerstrokecolor = :white, label = "adversarial ($adv_pred_class)",
)
plot!(
    p34, [x0[3], x_adv[3]], [x0[4], x_adv[4]],
    arrow = true, color = :gray, linewidth = 1.5, label = "",
)

fig = plot(p12, p34, layout = (1, 2), size = (1000, 400), margin = 5Plots.mm)
savefig(fig, joinpath(@__DIR__, "iris_bsr.svg")) #hide
fig #hide

# ## Common edits to try
#
# - Increase `epsilon` to make perturbations stronger.
# - Increase `max_iter` to give the attack more search time.
# - Adjust `bounds` to constrain or widen the search domain.
# - Attack other samples by changing how `demo_idx` is chosen.
# - Target a specific class by modifying the attack's objective function.
# - Change `Random.seed!` values to explore different search trajectories.
