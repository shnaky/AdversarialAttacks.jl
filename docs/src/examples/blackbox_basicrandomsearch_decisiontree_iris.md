# Black‑Box Basic Random Search (DecisionTree, Iris) — Step‑by‑step tutorial

This page walks you through running the example `examples/blackbox_basicrandomsearch_decisiontree_iris.jl` and interpreting its results.

**Prerequisites**
- Julia (compatible with the project's `Project.toml`, recommended version: 1.11).
- The project dependencies listed in `Project.toml`.

If you have the repository cloned, the simplest setup is:

```
# from the repository root
julia
julia> using Pkg
julia> Pkg.activate("./examples")
julia> Pkg.develop(path=".")
julia> Pkg.instantiate()
```

Once done you can run the example directly:

```
julia> include("examples/blackbox_basicrandomsearch_decisiontree_iris.jl")
```

Running the script shows plots and waits for Enter before exiting.

## What the script does

- **Load Iris** — reads the Iris dataset and prepares `X` (features) and `y_str` (string labels).
- **Train DecisionTree** — fits a small `DecisionTreeClassifier` and prints class information.
- **Pick demo sample** — selects a correctly classified sample and wraps it as `sample = (data, label)`.
- **Run BasicRandomSearch** — configures `BasicRandomSearch(epsilon, bounds, max_iter)` and calls `attack(atk, model, sample)`.
- **Evaluate & visualize** — compares original vs adversarial probabilities and plots 2D projections highlighting the two points.

## Interpreting the printed output

Key printed values and meaning:

Trained DecisionTreeClassifier on Iris.
Classes = ["setosa", "versicolor", "virginica"]

Chosen demo sample index: 51
Feature vector: [7.0, 3.2, 4.7, 1.4]
True label string: versicolor (index 2)

Original probabilities: [0.0 0.9791666666666666 0.020833333333333332]
Original predicted class index = CartesianIndex(1, 2)

Running BasicRandomSearch with epsilon = 0.3 and max_iter = 100 ...

Original feature vector:     Float32[7.0, 3.2, 4.7, 1.4]
Adversarial feature vector: Float32[7.0, 3.2, 5.0, 1.4]

Original probs:     [0.0 0.9791666666666666 0.020833333333333332]
Adversarial probs: [0.0 0.3333333333333333 0.6666666666666666]

True-class probability before attack: 0.9791666666666666
True-class probability after attack:  0.3333333333333333

[INFO] Attack decreased the true-class confidence (success).

Press Enter to close plots...



- **True label string**: The ground-truth class name (and its index) for the chosen sample.
- **Original probabilities / Original probs**: Model-predicted class probability vector for the original sample.
- **Original predicted class index**: The model's predicted class (as an index) for the original sample.
- **Adversarial feature vector**: The perturbed feature vector found by the attack.
- **Adversarial probs**: Model-predicted class probabilities for the adversarial sample.
- **True-class probability before attack / after attack**: The probability assigned to the true class before and after the attack. A drop indicates reduced confidence.

These metrics let you judge whether the attack both reduced model confidence and changed the final decision.

## Interpreting the plotted output

- The example produces two scatter plots (features 1&2 and 3&4). The original and adversarial samples are highlighted so you can see whether the attack moved the classification.
- The right panel includes an annotation with `true`, `orig`, and `adv` labels for quick inspection.

## Common edits to try

- Increase `epsilon` to make perturbations stronger.
- Increase `max_iter` to give the attack more time.
- Adjust `bounds` to constrain or widen the search domain.
- Attack other samples by changing how `demo_idx` is chosen.

## Troubleshooting

- If the attack rarely succeeds, raise `epsilon` or `max_iter`.
- If plots do not appear, replace `display(fig)` and `readline()` with `savefig(fig, "iris_bsr.png")` and inspect the saved image locally.

