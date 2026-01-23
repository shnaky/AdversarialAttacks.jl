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
    println(io, "  Clean-correct samples          : ",
        report.num_clean_correct, " / ", report.num_samples)

    println(io, "\nClean Performance")
    println(io, "  Clean accuracy                 : ",
        round(report.clean_accuracy * 100, digits=2), "%")

    println(io, "\nAdversarial Performance")
    println(io, "  Adversarial accuracy           : ",
        round(report.adv_accuracy * 100, digits=2), "%")

    println(io, "\nAttack Effectiveness")
    println(io, "  Successful attacks             : ",
        report.num_successful_attacks, " / ", report.num_clean_correct)
    println(io, "  Attack success rate (ASR)      : ",
        round(report.attack_success_rate * 100, digits=2), "%")
    println(io, "  Robustness score (1 - ASR)     : ",
        round(report.robustness_score * 100, digits=2), "%")

    println(io, "\nPerturbation Analysis (L_inf norm)")
    println(io, "  Maximum perturbation           : ",
        round(report.linf_norm_max, digits=6))
    println(io, "  Mean perturbation              : ",
        round(report.linf_norm_mean, digits=6))

    println(io, "\nNotes")
    println(io, "  • Attack success is counted only when:")
    println(io, "    - the clean prediction is correct")
    println(io, "    - the adversarial prediction is incorrect")
    println(io, "===================================")
end

"""
    evaluate_robustness(model, attack, test_data; num_samples=100)

Evaluate model robustness by running attack on multiple samples.

# Arguments
- `model`: The model to evaluate.
- `attack`: The attack to use.
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
    attack,
    test_data;
    num_samples::Int=100
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
    linf_norms = Float64[]

    for i in 1:n_test
        sample = test_data[i]
        true_label = argmax(sample.label)

        println("  Sample $i/$n_test")

        try
            # clean output
            clean_pred = model(sample.data)
            clean_label = argmax(clean_pred)
            is_clean_correct = (clean_label == true_label)
            num_clean_correct += is_clean_correct

            # adverserial output
            adv_data = craft(sample, model, attack)
            adv_pred = model(adv_data)
            adv_label = argmax(adv_pred)
            is_adv_correct = (adv_label == true_label)
            num_adv_correct += is_adv_correct

            # compute L_inf norm of perturbation
            perturbation = abs.(adv_data .- sample.data)
            linf_norm = maximum(perturbation)
            push!(linf_norms, linf_norm)

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

    # Metrics
    clean_accuracy = num_clean_correct / n_test
    adv_accuracy = num_adv_correct / n_test
    attack_success_rate = num_clean_correct > 0 ? num_successful_attacks / num_clean_correct : 0.0
    robustness_score = 1.0 - attack_success_rate
    linf_norm_max = length(linf_norms) > 0 ? maximum(linf_norms) : 0.0
    linf_norm_mean = length(linf_norms) > 0 ? sum(linf_norms) / length(linf_norms) : 0.0

    println("Evaluation complete!")

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
    )

end


