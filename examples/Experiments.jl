module Experiments

using MLJ
using MLJ: partition, accuracy
using MLDatasets
using StatsBase: mode
using DataFrames

export ExperimentConfig, run_experiment, load_mnist_for_mlj, flatten_images

# =========================
# Config
# =========================

struct ExperimentConfig
  name::String
  train_fraction::Float64
  rng::Int
end

const DEFAULT_CONFIG = ExperimentConfig("default", 0.8, 42)

# =========================
# Data loading
# =========================

"""
    flatten_images(X_img)

Convert a vector of H×W×C images into a DataFrame that can be used
with tabular MLJ models such as DecisionTreeClassifier.
Each pixel becomes one feature column.
"""
function flatten_images(X_img::Vector{<:AbstractArray})
  n = length(X_img)
  d = length(vec(X_img[1]))
  Xmat = Array{Float32}(undef, n, d)
  for i in 1:n
    Xmat[i, :] .= vec(X_img[i])
  end
  # wrap as a table with named columns
  df = DataFrame(Xmat, :auto)  # x1, x2, ... style column names
  return df
end


"""
    load_mnist_for_mlj(; n_train=60000)

Load the MNIST dataset and return `(X, y)` in a form that is
compatible with MLJ models.

- X: Vector of 28×28×1 Float32 arrays (one per image)
- y: CategoricalVector of digit labels 0–9
"""
function load_mnist_for_mlj(; n_train::Int=60000)
  # images: 28×28×N  UInt8
  # labels: Vector{Int}
  images, labels = MLDatasets.MNIST.traindata()
  images = images[:, :, 1:n_train]
  labels = labels[1:n_train]

  # convert to Float32 in [0, 1]
  X = Float32.(images) ./ 255

  # reshape to (H, W, C, N) if you later want MLJFlux.ImageClassifier
  # For now we just keep as a vector of (28,28,1) arrays:
  X_vec = [reshape(@view(X[:, :, i]), 28, 28, 1) for i in 1:size(X, 3)]

  # labels must be categorical for MLJ
  y = coerce(labels, Multiclass)

  return X_vec, y
end

# =========================
# Split + experiment
# =========================

"""
    train_test_split(n; fraction_train, rng)

Return `(train, test)` index vectors that partition `1:n` into
training and test sets using MLJ's `partition`.
"""
function train_test_split(n::Integer; fraction_train::Float64, rng::Int)
  train, test = partition(1:n, fraction_train, shuffle=true, rng=rng)
  return train, test
end

"""
    run_experiment(model, X, y; config=DEFAULT_CONFIG)

Train the given MLJ `model` on `(X, y)` using a simple train/test split
defined by `config`. Returns a named tuple with:

- mach:       the fitted MLJ machine
- train_idx:  indices of training samples
- test_idx:   indices of test samples
- ŷ_test:     predictions on the test set
- report:     small summary (e.g. accuracy)

This function does *not* perform any adversarial attacks; it only
sets up a clean baseline experiment that can later be reused by
attack/evaluation code.
"""
function run_experiment(model, X, y; config::ExperimentConfig=DEFAULT_CONFIG)
  n = length(y)
  train, test = train_test_split(n; fraction_train=config.train_fraction,
    rng=config.rng)

  # For DataFrame inputs we must use two-dimensional indexing
  Xtrain = X isa DataFrame ? X[train, :] : X[train]
  Xtest = X isa DataFrame ? X[test, :] : X[test]

  mach = machine(model, Xtrain, y[train])
  fit!(mach, verbosity=1)

  ŷ_test = predict(mach, Xtest)
  acc = MLJ.accuracy(mode.(ŷ_test), y[test])

  report = (accuracy=acc,)

  return (
    mach=mach,
    train_idx=train,
    test_idx=test,
    ŷ_test=ŷ_test,
    y_test=y[test],
    config=config,
    report=report,
  )
end


end
