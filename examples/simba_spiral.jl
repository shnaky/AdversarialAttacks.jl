# # Black-Box SimBA Attack on 2D Spirals
#
# This tutorial demonstrates how to perform a **black-box adversarial attack**
# using SimBA (Simple Black-box Attack) on a 2D spiral classification problem.
# BasicRandomSearch is our implementation of the SimBA algorithm.
#
# **What you will learn:**
# - How to create and train a model on a 2D spiral dataset
# - How to visualize decision boundaries and adversarial perturbations
# - How to use BasicRandomSearch (SimBA) for black-box attacks
# - How attack success varies with different epsilon values
#
# ## Prerequisites
#
# Make sure you have the following packages installed:
# `Flux`, `AdversarialAttacks`, `Plots`, `Statistics`, and `LinearAlgebra`.

using Random
using Flux
using AdversarialAttacks
using Plots
using Statistics
using LinearAlgebra: norm

Random.seed!(42)
println("=== SimBA Attack Demo ===\n")

# ## 1. Create 2D spiral dataset
#
# We generate a synthetic two-class dataset where each class forms a spiral
# pattern. This provides a challenging non-linear classification boundary that
# makes for interesting visualizations of adversarial perturbations.
function make_spirals(n_points = 100; noise = 0.3)
    t = range(0, 4π, length = n_points)

    ## Class 1: spiral going one way
    x1 = t .* cos.(t) .+ noise .* randn(n_points)
    y1 = t .* sin.(t) .+ noise .* randn(n_points)

    ## Class 2: spiral going the other way
    x2 = t .* cos.(t .+ π) .+ noise .* randn(n_points)
    y2 = t .* sin.(t .+ π) .+ noise .* randn(n_points)

    X = hcat(vcat(x1, x2), vcat(y1, y2))'  # 2 x 2n
    y = vcat(ones(Int, n_points), 2 * ones(Int, n_points))

    ## Normalize
    X = (X .- mean(X, dims = 2)) ./ std(X, dims = 2)

    return Float32.(X), y
end

println("Generating spiral dataset...")
X, y = make_spirals(150; noise = 0.4)

# ## 2. Train simple neural network
#
# We train a small feedforward neural network (2→16→16→2) on the spiral data.
# The model learns to separate the two spirals using ReLU activations and
# cross-entropy loss.
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

# ## 3. Visualization helpers
#
# These functions help us visualize the decision boundary and attack results.
# The `plot_decision_boundary!` function creates a contour plot showing which
# regions the model predicts as class 1 vs class 2.
function plot_decision_boundary!(plt, model; resolution = 100, alpha = 0.3)
    xs = range(-3, 3, length = resolution)
    ys = range(-3, 3, length = resolution)

    Z = zeros(resolution, resolution)
    for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
        pred = model(Float32[x, y])
        Z[j, i] = pred[1] - pred[2]
    end

    ## Replace NaN/Inf values to prevent plotting errors
    Z = replace(Z, NaN => 0.0, Inf => 10.0, -Inf => -10.0)

    return contourf!(
        plt, xs, ys, Z, levels = [-10, 0, 10],
        c = cgrad([:lightsalmon, :lightblue]), alpha = alpha,
        linewidth = 0, colorbar = false
    )
end

function plot_attack_results(X, y, model, atk; n_samples = 20)
    indices = randperm(size(X, 2))[1:n_samples]

    plt = plot(
        size = (800, 700), title = "SimBA Attack (ε=$(atk.epsilon))",
        xlabel = "x₁", ylabel = "x₂", legend = :topright
    )
    plot_decision_boundary!(plt, model)

    successful_attacks = 0
    total_perturbation = 0.0

    for idx in indices
        x_orig = X[:, idx]
        label_onehot = Flux.onehot(y[idx], 1:2)
        sample = (data = x_orig, label = label_onehot)

        x_adv = attack(atk, model, sample)

        ## Skip if attack produced NaN or Inf values
        if any(isnan.(x_adv)) || any(isinf.(x_adv))
            continue
        end

        pred_orig = argmax(model(x_orig))
        pred_adv = argmax(model(x_adv))

        success = pred_orig != pred_adv
        if success
            successful_attacks += 1
        end
        total_perturbation += norm(x_adv - x_orig)

        ## Plot original point
        color = y[idx] == 1 ? :blue : :red
        scatter!(
            plt, [x_orig[1]], [x_orig[2]],
            color = color, markersize = 8, markerstrokewidth = 2,
            label = (idx == indices[1] ? "Original (class $(y[idx]))" : "")
        )

        ## Plot adversarial point
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

        ## Draw arrow from original to adversarial
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

    bounds = [(-3.5, 3.5), (-3.5, 3.5)]

    for ε in epsilons
        atk = BasicRandomSearch(epsilon = Float32(ε), max_iter = 100, bounds = bounds)
        plt = plot_attack_results(X, y, model, atk; n_samples = n_samples)
        title!(plt, "ε = $ε")
        push!(plots, plt)
    end

    combined = plot(plots..., layout = (2, 2), size = (1200, 1100))
    return combined
end

# ## 4. Run attack and visualize
#
# We run the BasicRandomSearch attack with ε=0.5 and visualize the results.
# The attack tries to find small perturbations that cause misclassification
# by randomly probing the input space. Original points are shown in their
# class color (blue/red), successful adversarial examples as black X markers,
# and failed attacks as gray circles.
println("\nRunning SimBA attack visualization...")
bounds = [(-3.5, 3.5), (-3.5, 3.5)]  # Set bounds for normalized data
atk = BasicRandomSearch(epsilon = 0.5f0, max_iter = 100, bounds = bounds)
p1 = plot_attack_results(X, y, model, atk; n_samples = 25)

savefig(p1, joinpath(@__DIR__, "simba_single.svg")) #hide
p1 #hide

# ## 5. Compare different epsilon values
#
# We compare attack success rates across different perturbation budgets (ε).
# Larger ε values allow stronger perturbations, making attacks more likely to
# succeed but also more visible. The grid layout shows how attack effectiveness
# scales with perturbation budget.

println("\nComparing different epsilon values...")
p2 = compare_epsilons(X, y, model; epsilons = [0.1, 0.3, 0.5, 1.0], n_samples = 25)

savefig(p2, joinpath(@__DIR__, "simba_epsilons.svg")) #hide
p2 #hide

# ## Common edits to try
#
# - Change `epsilon` values to see how perturbation budget affects attack success
# - Adjust `max_iter` to give the attack more or fewer queries
# - Modify `noise` in `make_spirals()` to change problem difficulty
# - Try different network architectures by changing the Dense layer sizes
# - Change `n_samples` to attack more or fewer points
