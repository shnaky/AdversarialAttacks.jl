using Random
using RDatasets
using DecisionTree
using Flux
using OneHotArrays: OneHotVector
using AdversarialAttacks
using Plots

Random.seed!(1234)

# ------------------------------------
# 1. Train DecisionTree on Iris
# ------------------------------------
iris = dataset("datasets", "iris")
X = Matrix{Float64}(iris[:, 1:4])
y_str = String.(iris.Species)
classes = ["setosa", "versicolor", "virginica"]

dt_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=1,
    min_samples_split=2,
    classes=classes,
)
fit!(dt_model, X, y_str)

println("Trained DecisionTreeClassifier on Iris.")
println("Classes = ", dt_model.classes)

# helper
function predict_class_index(model::DecisionTreeClassifier, x::AbstractVector)
    x_mat = reshape(Float64.(x), 1, :)
    probs = DecisionTree.predict_proba(model, x_mat)
    return argmax(probs)
end

# ------------------------------------
# 2. Pick first correctly classified sample
# ------------------------------------
demo_idx = findfirst(==("versicolor"), y_str)

for i in 1:size(X, 1)
    xi = X[i, :]
    yi_str = y_str[i]
    true_idx_i = findfirst(==(yi_str), classes)
    pred_idx_i = predict_class_index(dt_model, xi)
    if pred_idx_i == true_idx_i
        demo_idx = i
        break
    end
end

x0 = X[demo_idx, :]
label_str = y_str[demo_idx]
true_idx = findfirst(==(label_str), classes)

println("\nChosen demo sample index: ", demo_idx)
println("Feature vector: ", x0)
println("True label string: ", label_str, " (index ", true_idx, ")")

# ------------------------------------
# 3. Build sample NamedTuple
# ------------------------------------
y0 = Flux.onehot(true_idx, 1:length(classes))
sample = (data=Float32.(x0), label=y0)

x0_mat = reshape(Float64.(x0), 1, :)
orig_probs_vec = DecisionTree.predict_proba(dt_model, x0_mat)
orig_true_prob = orig_probs_vec[true_idx]

println("\nOriginal probabilities: ", orig_probs_vec)
println("Original predicted class index = ", argmax(orig_probs_vec))

# ------------------------------------
# 4. Run BasicRandomSearch
# ------------------------------------
ε = 0.3f0
attack = BasicRandomSearch(
    ε,
    [(4.3f0, 7.9f0), (2.0f0, 4.4f0), (1.0f0, 6.9f0), (0.1f0, 2.5f0)]
)
println("\nRunning BasicRandomSearch with epsilon = ", ε, " ...")
Random.seed!(42)

x_adv = craft(sample, dt_model, attack)

x_adv_mat = reshape(Float64.(x_adv), 1, :)
adv_probs_vec = DecisionTree.predict_proba(dt_model, x_adv_mat)
adv_true_prob = adv_probs_vec[true_idx]

println("\nOriginal feature vector:     ", sample.data)
println("Adversarial feature vector: ", x_adv)

println("\nOriginal probs:     ", orig_probs_vec)
println("Adversarial probs: ", adv_probs_vec)

println("\nTrue-class probability before attack: ", orig_true_prob)
println("True-class probability after attack:  ", adv_true_prob)

if adv_true_prob < orig_true_prob
    println("\n[INFO] Attack decreased the true-class confidence (success).")
else
    println("\n[INFO] True-class confidence did not decrease.")
end

# ------------------------------------
# 5. Visualization: 2D projections of Iris + highlighted sample
# ------------------------------------
idx_setosa = findall(==("setosa"), y_str)
idx_versicolor = findall(==("versicolor"), y_str)
idx_virginica = findall(==("virginica"), y_str)

# Plot 1: features 1 & 2
p12 = plot(xlabel="SepalLength", ylabel="SepalWidth", title="Iris (features 1&2)", legend=false)
scatter!(p12, X[idx_setosa, 1], X[idx_setosa, 2], color=:blue, markershape=:circle)
scatter!(p12, X[idx_versicolor, 1], X[idx_versicolor, 2], color=:green, markershape=:utriangle)
scatter!(p12, X[idx_virginica, 1], X[idx_virginica, 2], color=:red, markershape=:square)
scatter!(p12, [x0[1]], [x0[2]], markersize=10, color=:black, label="")
scatter!(p12, [x_adv[1]], [x_adv[2]], markersize=10, color=:orange, label="")

# Plot 2: features 3 & 4
orig_pred_class = classes[argmax(orig_probs_vec)[2]]
adv_pred_class = classes[argmax(adv_probs_vec)[2]]
p34 = plot(xlabel="PetalLength", ylabel="PetalWidth", title="Iris (features 3&4)")
scatter!(p34, X[idx_setosa, 3], X[idx_setosa, 4], color=:blue, markershape=:circle, label="setosa")
scatter!(p34, X[idx_versicolor, 3], X[idx_versicolor, 4], color=:green, markershape=:utriangle, label="versicolor")
scatter!(p34, X[idx_virginica, 3], X[idx_virginica, 4], color=:red, markershape=:square, label="virginica")
scatter!(p34, [x0[3]], [x0[4]], markersize=10, color=:black, label="original")
scatter!(p34, [x_adv[3]], [x_adv[4]], markersize=10, color=:orange, label="adversarial")

annot_str = "true: $label_str\norig: $orig_pred_class\nadv: $adv_pred_class"
annotate!(p34, x0[3] + 0.2, x0[4], text(annot_str, 8, :left))

fig = plot(p12, p34, layout=(1, 2))
display(fig)

println("\nPress Enter to close plots...")
readline()
