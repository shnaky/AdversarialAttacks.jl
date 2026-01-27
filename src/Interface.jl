"""
    benchmark(atk::AbstractAttack, model, dataset, metric::Function; kwargs...)

Evaluate attack performance on a dataset with labels using a given metric.

# Arguments
- `atk::AbstractAttack`: Attack algorithm
- `model`: Target model to attack
- `dataset`: Dataset with samples and labels
- `metric::Function`: Evaluation metric with signature `metric(model, adv_samples, labels)`

# Returns
- Scalar metric value representing attack performance
"""
function benchmark(atk::AbstractAttack, model, dataset, metric::Function; kwargs...)
    adv_samples = [attack(atk, model, x; kwargs...) for (x, _) in dataset]
    labels = [y for (_, y) in dataset]
    return metric(model, adv_samples, labels)
end


