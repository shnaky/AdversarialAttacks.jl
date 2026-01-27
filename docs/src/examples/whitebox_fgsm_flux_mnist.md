# White‑Box FGSM (Flux, MNIST) — Step‑by‑step tutorial

This page walks you through running and understanding the example script `examples/whitebox_fgsm_flux_mnist.jl`. It shows how to run the example, where to inspect results, and how to adapt common parameters.

**Prerequisites**
- Julia (compatible with the project's `Project.toml`, recommended version: 1.11).
- The project dependencies listed in `Project.toml` (this repository provides them).

If you have the repository cloned, the simplest setup is:

```
# from the repository root
julia
julia> using Pkg
julia> Pkg.add.( ["Flux", "MLDatasets", "Images"] )
julia> Pkg.activate("./examples")
julia> Pkg.develop(path=".")
julia> Pkg.instantiate()
```

This activates the local project and installs the exact dependencies pinned in the repo. Once done you can run the example directly:

```
julia> include("examples/whitebox_fgsm_flux_mnist.jl")
```

Running the script shows plots and waits for Enter before exiting.

### What the script does

- **Load MNIST subset** — loads a subset of `MLDatasets.MNIST` and normalizes pixels to $[0,1]$`. The input tensor  `X`  has shape `(28,28,1,N)` and `y` is a one‑hot encoded label matrix.
- **Define and train CNN** — constructs a small Flux `Chain`, defines `loss` and `opt`, and runs a few epochs over mini‑batches. This trains a model on the small subset to make the demo deterministic and fast.
- **Pick a demo sample** — selects a single correctly classified sample (`demo_idx`) to attack. The sample is put into the tuple `sample = (data = x0, label = label_onehot)` which is the expected input format for attacks in this package.
- **Run FGSM attack** — constructs `FGSM(epsilon = ε)` and calls `attack(fgsm_attack, model, sample)`. The returned `x_adv` is clamped to `[0,1]`.
- **Evaluate and report** — computes the original and adversarial predictions, prints probabilities, computes the perturbation L∞ norm `maximum(abs.(x_adv .- x0))`, and prints simple booleans for success criteria.
- **Visualization** — three heatmaps: original image, adversarial image, and the perturbation (difference). The script displays the figures and waits for user input.

## Interpreting the printed output

Key printed values and what they mean:


=== White-Box FGSM Attack Tutorial ===

✓ Trained simple CNN on MNIST subset

Chosen sample index: 2
True digit: 0  (index=1)
Clean prediction: 0  (index=1)
Clean probs: Float32[0.979; 0.0; 0.0; 0.002; 0.0; 0.018; 0.0; 0.0; 0.001; 0.0;;]
Clean true prob: 0.979

Running FGSM with ε = 0.0015 ...

Original image stats   : min=0.0, max=0.003921569
Adversarial image stats: min=0.0, max=0.0054215686
Perturbation L∞ norm   : 0.0015000002

Adversarial probs: Float32[0.0; 0.0; 0.0; 0.04; 0.0; 0.026; 0.0; 0.914; 0.005; 0.015;;]
True prob: 0.979 → 0.0
[INFO] True-class prob drop success: true  (0.979 → 0.0)
[INFO] Prediction flip success: true  (clean_digit=0, adv_digit=7)
Digits summary: true=0, clean=0, adv=7

Press Enter to exit...



- **Train-subset acc:** rough sanity check showing the model learned useful features.
- **Clean probs / Adversarial probs:** the softmax output giving class probabilities.
- **Perturbation L∞ norm:** the maximum absolute pixel change.
- **True-class prob drop success:** `true` if the true label probability decreased.
- **Prediction flip success:** `true` if the predicted label changed from the original to the adversarial example.

These metrics let you judge whether the attack both reduced model confidence and changed the final decision.

### Interpreting the plotted output

The effect of the attack is shown in the following figure, included in the documentation as:

![FGSM attack on MNIST](../assets/mnist_fgsm.png)
The three panels show:

- **Original (digit = 0)**: The clean MNIST image that the model classifies correctly.
- **Adversarial (digit = 7)**: The perturbed image after the FGSM attack, which the model now misclassifies as a different digit.
- **Perturbation ($\epsilon = 0.0015$)**: A heatmap of the pixel‑wise difference, showing that only a thin border of pixels around the digit is changed while the overall shape stays visually similar.


## Common edits to try

- Change the attack parameter `ε` (e.g., `0.0015f0 → 0.003f0`) to make perturbations stronger or weaker. Higher `ε` corresponds to stronger disturbance. Expect larger L∞ values and higher flip success.
- Change `demo_idx` to attack different digits. Some digits are easier to flip than others.
- Increase training `epochs` or use more samples to get a stronger base classifier (edit the `for epoch in 1:5` loop).
- Reduce `batch_size` or use fewer training samples to speed up the demo on slow machines.

## Troubleshooting

- Slow training: reduce the training subset size or lower `epochs`. The example intentionally trains only 6000 samples for speed.
- Plot not showing in some remote environments: save the figure with `savefig(...)` and inspect it locally.
