# White-Box – FGSM (Flux, MNIST)

This tutorial illustrates a **white‑box** attack using the Fast Gradient Sign Method (FGSM) against a convolutional neural network trained on MNIST handwritten digits. The goal is to show how small, gradient‑based perturbations can flip the model’s prediction while remaining almost imperceptible.

### What the script does

- Loads a subset of MNIST, normalizes pixel values to $[0, 1]$, and trains a simple CNN with Flux.
- Wraps the trained a CNN model in `Flux` and crafts an adversarial example for a correctly classified digit using FGSM with a small $L_{\infty}$ budget.
- Reports clean vs adversarial probabilities, the drop in true‑class confidence, and whether the predicted digit changes.


### Visual summary

The effect of the attack is summarized by the following figure, included in the documentation as:

![FGSM attack on MNIST](../assets/mnist_fgsm.png)

The three panels show:

- **Original (digit = 0)**: The clean MNIST image that the model classifies correctly.
- **Adversarial (digit = 7)**: The perturbed image after the FGSM attack, which the model now misclassifies as a different digit.
- **Perturbation ($\epsilon = 0.0015$)**: A heatmap of the pixel‑wise difference, highlighting that only a thin band of pixels around the digit is modified while the overall shape remains visually similar.

This figure makes the white‑box threat model and the subtlety of the perturbation immediately apparent.