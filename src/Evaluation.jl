"""
    Robustness evaluation suite for adversarial attacks.

    This module provides functions to evaluate model robustness
    by measuring attack success rates on multiple samples.
"""

"""
    RobustnessReport

Report on model robustness against an adversarial attack.
Printing a `RobustnessReport` (via `println(report)`) displays a nicely formatted summary
including clean/adversarial accuracy, attack success rate, and robustness score.

# Fields
- `num_samples::Int`: Total samples evaluated
- `num_clean_correct::Int`: Samples correctly classified before attack
- `clean_accuracy::Float64`: Accuracy on clean samples
- `adv_accuracy::Float64`: Accuracy on adversarial samples
- `attack_success_rate::Float64`: (ASR) Fraction of successful attacks (on correctly classified samples)
- `robustness_score::Float64`: 1.0 - attack_success_rate (ASR)
- `num_successful_attacks::Int`: Number of successful attacks
- `linf_norm_max::Float64`: Maximum L_inf norm of perturbations across all samples
- `linf_norm_mean::Float64`: Mean L_inf norm of perturbations across all samples

# Note
An attack succeeds when the clean prediction is correct but the adversarial prediction is incorrect.
The L_inf norm measures the maximum absolute change in any feature of the input.
"""
struct RobustnessReport
    num_samples::Int
    num_clean_correct::Int
    clean_accuracy::Float64
    adv_accuracy::Float64
    attack_success_rate::Float64
    robustness_score::Float64
    num_successful_attacks::Int
    linf_norm_max::Float64
    linf_norm_mean::Float64
    l2_norm_max::Float64
    l2_norm_mean::Float64
end

"""
    Base.show(io::IO, report::RobustnessReport)

Internal method that defines how a `RobustnessReport` is displayed
when printed. Shows dataset statistics, clean and adversarial accuracy,
attack success rate, and robustness score in a formatted report.

Typically called automatically by `println(report)`.
"""

function Base.show(io::IO, report::RobustnessReport)
    println(io, "=== Robustness Evaluation Report ===")

    println(io, "\nDataset")
    println(io, "  Total samples evaluated        : ", report.num_samples)
    println(
        io, "  Clean-correct samples          : ",
        report.num_clean_correct, " / ", report.num_samples
    )

    println(io, "\nClean Performance")
    println(
        io, "  Clean accuracy                 : ",
        round(report.clean_accuracy * 100, digits = 2), "%"
    )

    println(io, "\nAdversarial Performance")
    println(
        io, "  Adversarial accuracy           : ",
        round(report.adv_accuracy * 100, digits = 2), "%"
    )

    println(io, "\nAttack Effectiveness")
    println(
        io, "  Successful attacks             : ",
        report.num_successful_attacks, " / ", report.num_clean_correct
    )
    println(
        io, "  Attack success rate (ASR)      : ",
        round(report.attack_success_rate * 100, digits = 2), "%"
    )
    println(
        io, "  Robustness score (1 - ASR)     : ",
        round(report.robustness_score * 100, digits = 2), "%"
    )

    println(io, "\nPerturbation Analysis (Norms)")
    println(
        io, "  L_inf Maximum perturbation           : ",
        round(report.linf_norm_max, digits = 2)
    )
    println(
        io, "  L_inf Mean perturbation              : ",
        round(report.linf_norm_mean, digits = 2)
    )
    println(
        io, "  L_2 Max perturbation                 : ",
        round(report.l2_norm_max, digits = 2)
    )
    println(
        io, "  L_2 Mean perturbation                : ",
        round(report.l2_norm_max, digits = 2)
    )

    println(io, "\nNotes")
    println(io, "  • Attack success is counted only when:")
    println(io, "    - the clean prediction is correct")
    println(io, "    - the adversarial prediction is incorrect")
    return println(io, "===================================")
end


"""
    calculate_metrics(n_test, num_clean_correct, num_adv_correct,
                      num_successful_attacks, l_norms)

Compute accuracy, attack success, robustness, and perturbation norm statistics
for adversarial evaluation.

# Arguments
- `n_test`: Number of test samples.
- `num_clean_correct`: Number of correctly classified clean samples.
- `num_adv_correct`: Number of correctly classified adversarial samples.
- `num_successful_attacks`: Number of successful adversarial attacks.
- `l_norms`: Dictionary containing perturbation norms (e.g. `:linf`, `:l2`).

# Returns
- A `RobustnessReport` containing accuracy, robustness, and norm summary metrics.
"""
function calculate_metrics(n_test, num_clean_correct, num_adv_correct, num_successful_attacks, l_norms)

    clean_accuracy = num_clean_correct / n_test
    adv_accuracy = num_adv_correct / n_test
    attack_success_rate = num_clean_correct > 0 ? num_successful_attacks / num_clean_correct : 0.0
    robustness_score = 1.0 - attack_success_rate
    # norms
    linf_norms = l_norms[:linf]
    linf_norm_max = length(linf_norms) > 0 ? maximum(linf_norms) : 0.0
    linf_norm_mean = length(linf_norms) > 0 ? sum(linf_norms) / length(linf_norms) : 0.0

    l2_norms = l_norms[:l2]
    l2_norm_max = !isempty(l2_norms) ? maximum(l2_norms) : 0.0
    l2_norm_mean = !isempty(l2_norms) ? sum(l2_norms) / length(l2_norms) : 0.0

    return RobustnessReport(
        n_test,
        num_clean_correct,
        clean_accuracy,
        adv_accuracy,
        attack_success_rate,
        robustness_score,
        num_successful_attacks,
        linf_norm_max,
        linf_norm_mean,
        l2_norm_max,
        l2_norm_mean,
    )
end

"""
    compute_norm(sample_data, adv_data, p::Real)

Compute the Lp norm of the perturbation between original data and adversarial data.

# Arguments
- `sample_data`: Original sample data.
- `adv_data`: Adversarially perturbed version of `sample_data`.
- `p::Real`: Order of the norm (`p > 0` or `Inf`).

# Returns
- The Lp norm between the two input data arrays.
"""
function compute_norm(sample_data, adv_data, p::Real)
    # source: https://en.wikipedia.org/wiki/Lp_space

    perturbation = adv_data .- sample_data

    if p == Inf
        return maximum(abs.(perturbation))
    elseif p > 0
        return sum(abs.(perturbation) .^ p)^(1 / p)
    else
        error("Unsupported norm p=$p; must be positive or Inf")
    end
end


"""
    evaluate_robustness(model, atk, test_data; num_samples=100)

Evaluate model robustness by running attack on multiple samples.

# Arguments
- `model`: The model to evaluate.
- `atk`: The attack to use.
- `test_data`: Collection of test samples.
- `num_samples::Int=100`: Number of samples to test. If more than available samples,
  uses all available samples.

# Returns
- `RobustnessReport`: Report containing accuracy, attack success rate, and robustness metrics

# Example
```julia
report = evaluate_robustness(model, FGSM(ε=0.1), test_data, num_samples=50)
println(report)
```
"""
function evaluate_robustness(
        model,
        atk,
        test_data;
        num_samples::Int = 100
    )
    if num_samples <= 0
        throw(ArgumentError("num_samples must be positive"))
    end

    n_available = length(test_data)
    n_test = min(num_samples, n_available)

    println("Testing $n_test samples...")

    # aggregators
    num_clean_correct = 0
    num_adv_correct = 0
    num_successful_attacks = 0
    l_norms = Dict(
        :linf => Float64[],
        :l2 => Float64[]
    )

    predict_fn = make_prediction_function(model)

    for i in 1:n_test
        sample = test_data[i]
        true_label = argmax(sample.label)

        println("  Sample $i/$n_test")

        try
            # clean output
            clean_pred = predict_fn(sample.data)
            clean_label = argmax(vec(clean_pred))
            is_clean_correct = (clean_label == true_label)
            num_clean_correct += is_clean_correct

            # adverserial output
            adv_data = attack(atk, model, sample)
            adv_pred = predict_fn(adv_data)
            adv_label = argmax(vec(adv_pred))
            is_adv_correct = (adv_label == true_label)
            num_adv_correct += is_adv_correct

            push!(l_norms[:linf], compute_norm(sample.data, adv_data, Inf))
            push!(l_norms[:l2], compute_norm(sample.data, adv_data, 2))

            # successful attack condition (a flip happened in prediction)
            if is_clean_correct && !is_adv_correct
                num_successful_attacks += 1
            end
        catch e
            @warn "Failed to evaluate sample $i" exception = e
        end

        if i % 10 == 0 || i == n_test
            println("  Processed $i/$n_test samples...")
        end
    end

    report = calculate_metrics(n_test, num_clean_correct, num_adv_correct, num_successful_attacks, l_norms)
    println("Evaluation complete!")
    return report
end

"""
    evaluation_curve(model, atk_type, epsilons, test_data; num_samples=100)

Evaluate model robustness across a range of attack strengths.

For each value in `epsilons`, an attack of type `atk_type` is instantiated and
used to compute clean accuracy, adversarial accuracy, attack success rate, and
perturbation norms.

# Arguments
- `model`: Model to be evaluated.
- `atk_type`: Adversarial attack type.
- `epsilons`: Vector of attack strengths.
- `test_data`: Test dataset.

# Keyword Arguments
- `num_samples`: Number of samples used for each evaluation.

# Returns
- A dictionary containing evaluation metrics indexed by `epsilons`.
"""
function evaluation_curve(model, atk_type::Type{<:AbstractAttack}, epsilons::Vector{Float64}, test_data; num_samples::Int = 100)
    results = Dict(
        :epsilons => Float64[],
        :clean_accuracy => Float64[],
        :adv_accuracy => Float64[],
        :attack_success_rate => Float64[],
        :linf_norm_mean => Float64[]
    )

    for epsilon in epsilons
        # TODO: there should be a set rng parameter for BSR so it's ther results can be compared
        atk = atk_type(epsilon)

        report = evaluate_robustness(
            model,
            atk,
            test_data;
            num_samples = num_samples
        )

        push!(results[:epsilons], epsilon)
        push!(results[:clean_accuracy], report.clean_accuracy)
        push!(results[:adv_accuracy], report.adv_accuracy)
        push!(results[:attack_success_rate], report.attack_success_rate)
        push!(results[:linf_norm_mean], report.linf_norm_mean)
    end

    return results
end

"""
    make_prediction_function(model)

Create a unified prediction function for evaluation only.
This is NOT passed to attack() - only used for getting predictions.
"""

function make_prediction_function(model)
    if model isa Machine
        return function (x_data)
            if x_data isa AbstractVector
                x_row = reshape(x_data, 1, :)
            else
                x_row = reshape(vec(x_data), 1, :)
            end

            X_tbl = table(x_row)
            pred_dist = predict(model, X_tbl)[1]

            class_levels = levels(pred_dist)
            probs = [pdf(pred_dist, level) for level in class_levels]

            return probs
        end
    else
        return function (x)
            output = model(x)
            return vec(output)
        end
    end
end
