# SimBA (BasicRandomSearch) Demo
# Visualizes the attack on a 2D classification problem

using Random
using Flux
using AdversarialAttacks
using Plots
using Statistics
using LinearAlgebra: norm

Random.seed!(42)
println("=== SimBA Attack Demo ===\n")

# ------------------------------------
# 1. Create 2D spiral dataset
# ------------------------------------
function make_spirals(n_points = 100; noise = 0.3)
    t = range(0, 4π, length = n_points)

    # Class 1: spiral going one way
    x1 = t .* cos.(t) .+ noise .* randn(n_points)
    y1 = t .* sin.(t) .+ noise .* randn(n_points)

    # Class 2: spiral going the other way
    x2 = t .* cos.(t .+ π) .+ noise .* randn(n_points)
    y2 = t .* sin.(t .+ π) .+ noise .* randn(n_points)

    X = hcat(vcat(x1, x2), vcat(y1, y2))'  # 2 x 2n
    y = vcat(ones(Int, n_points), 2 * ones(Int, n_points))

    # Normalize
    X = (X .- mean(X, dims = 2)) ./ std(X, dims = 2)

    return Float32.(X), y
end

println("Generating spiral dataset...")
X, y = make_spirals(150; noise = 0.4)

# ------------------------------------
# 2. Train simple neural network
# ------------------------------------
function train_model(X, y; epochs = 500)
    model = Chain(
        Dense(2 => 16, relu),
        Dense(16 => 16, relu),
        Dense(16 => 2),
    )

    Y_onehot = Flux.onehotbatch(y, 1:2)
    opt = Flux.setup(Adam(0.01), model)

    for epoch in 1:epochs
        loss, grads = Flux.withgradient(model) do m
            Flux.logitcrossentropy(m(X), Y_onehot)
        end
        Flux.update!(opt, model, grads[1])

        if epoch % 100 == 0
            acc = mean(Flux.onecold(model(X)) .== y)
            println("Epoch $epoch: loss = $(round(loss, digits = 4)), acc = $(round(acc, digits = 3))")
        end
    end

    return model
end

println("\nTraining neural network...")
model = train_model(X, y; epochs = 500)

final_acc = mean(Flux.onecold(model(X)) .== y)
println("Final accuracy: $(round(100 * final_acc, digits = 1))%")

# ------------------------------------
# 3. Visualization helpers
# ------------------------------------
function plot_decision_boundary!(plt, model; resolution = 100, alpha = 0.3)
    xs = range(-3, 3, length = resolution)
    ys = range(-3, 3, length = resolution)

    Z = zeros(resolution, resolution)
    for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
        pred = model([x, y])
        Z[j, i] = pred[1] - pred[2]
    end

    return contourf!(
        plt, xs, ys, Z, levels = [-10, 0, 10],
        c = cgrad([:lightblue, :lightsalmon]), alpha = alpha,
        linewidth = 0, colorbar = false
    )
end

function plot_attack_results(X, y, model, attack; n_samples = 20)
    indices = randperm(size(X, 2))[1:n_samples]

    plt = plot(
        size = (800, 700), title = "SimBA Attack (ε=$(attack.epsilon))",
        xlabel = "x₁", ylabel = "x₂", legend = :topright
    )
    plot_decision_boundary!(plt, model)

    successful_attacks = 0
    total_perturbation = 0.0

    for idx in indices
        x_orig = X[:, idx]
        label_onehot = Flux.onehot(y[idx], 1:2)
        sample = (data = x_orig, label = label_onehot)

        x_adv = craft(sample, model, attack)

        pred_orig = argmax(model(x_orig))
        pred_adv = argmax(model(x_adv))

        success = pred_orig != pred_adv
        if success
            successful_attacks += 1
        end
        total_perturbation += norm(x_adv - x_orig)

        # Plot original point
        color = y[idx] == 1 ? :blue : :red
        scatter!(
            plt, [x_orig[1]], [x_orig[2]],
            color = color, markersize = 8, markerstrokewidth = 2,
            label = (idx == indices[1] ? "Original (class $(y[idx]))" : "")
        )

        # Plot adversarial point
        if success
            scatter!(
                plt, [x_adv[1]], [x_adv[2]],
                color = :black, marker = :x, markersize = 10, markerstrokewidth = 3,
                label = (successful_attacks == 1 ? "Adversarial (misclassified)" : "")
            )
        else
            scatter!(
                plt, [x_adv[1]], [x_adv[2]],
                color = :gray, marker = :circle, markersize = 5, alpha = 0.5,
                label = ""
            )
        end

        # Draw arrow from original to adversarial
        arrow_color = success ? :black : :gray
        plot!(
            plt, [x_orig[1], x_adv[1]], [x_orig[2], x_adv[2]],
            color = arrow_color, alpha = 0.5, linewidth = 1, arrow = true, label = ""
        )
    end

    attack_rate = round(100 * successful_attacks / n_samples, digits = 1)
    avg_pert = round(total_perturbation / n_samples, digits = 3)
    annotate!(plt, -2.8, 2.8, text("Attack success: $attack_rate%\nAvg perturbation: $avg_pert", 10, :left))

    return plt
end

function compare_epsilons(X, y, model; epsilons = [0.1, 0.3, 0.5, 1.0], n_samples = 30)
    plots = []

    for ε in epsilons
        attack = BasicRandomSearch(epsilon = Float32(ε))
        plt = plot_attack_results(X, y, model, attack; n_samples = n_samples)
        title!(plt, "ε = $ε")
        push!(plots, plt)
    end

    combined = plot(plots..., layout = (2, 2), size = (1200, 1100))
    return combined
end

# ------------------------------------
# 4. Run attack and visualize
# ------------------------------------
println("\nRunning SimBA attack visualization...")
attack = BasicRandomSearch(epsilon = 0.5f0)
p1 = plot_attack_results(X, y, model, attack; n_samples = 25)
display(p1)

println("\nComparing different epsilon values...")
p2 = compare_epsilons(X, y, model; epsilons = [0.1, 0.3, 0.5, 1.0], n_samples = 25)
display(p2)

println("\nPress Enter to exit...")
readline()
