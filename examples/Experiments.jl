module Experiments

using MLJ
using MLJ: partition, accuracy
using MLJFlux
using Flux
using Optimisers

using MLDatasets
using StatsBase: mode
using DataFrames

using ColorTypes: Gray
using Images

export ExperimentConfig, run_experiment, load_mnist_for_mlj, flatten_images
export make_mnist_cnn

using ColorTypes: Gray
using Images
export make_mnist_forest, make_mnist_tree
export blackbox_predict
export make_mnist_cnn

const DecisionTreeClassifier = @load DecisionTreeClassifier pkg = DecisionTree
const RandomForestClassifier = @load RandomForestClassifier pkg = DecisionTree

const ImageClassifier = MLJFlux.ImageClassifier

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
compatible with MLJFlux.ImageClassifier.

- X: Vector of 28×28 `Gray` images (one per sample)
- y: CategoricalVector of digit labels 0–9
"""
function load_mnist_for_mlj(; n_train::Int = 60000)
    # images: 28×28×N  UInt8
    # labels: Vector{Int}
    images, labels = MLDatasets.MNIST(split = :train)[:]
    images = images[:, :, 1:n_train]
    labels = labels[1:n_train]

    # convert to Gray images in [0, 1]
    X_vec = [Gray.(images[:, :, i] ./ 255) for i in 1:n_train]

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
    train, test = partition(1:n, fraction_train, shuffle = true, rng = rng)
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
function run_experiment(model, X, y; config::ExperimentConfig = DEFAULT_CONFIG)
    n = length(y)
    train, test =
        train_test_split(n; fraction_train = config.train_fraction, rng = config.rng)

    # For DataFrame inputs we must use two-dimensional indexing
    Xtrain = X isa DataFrame ? X[train, :] : X[train]
    Xtest = X isa DataFrame ? X[test, :] : X[test]

    mach = machine(model, Xtrain, y[train])
    fit!(mach, verbosity = 1)

    ŷ_test = predict(mach, Xtest)
    acc = accuracy(mode.(ŷ_test), y[test])

    report = (accuracy = acc,)

    return (
        mach = mach,
        train_idx = train,
        test_idx = test,
        ŷ_test = ŷ_test,
        y_test = y[test],
        config = config,
        report = report,
    )
end

"""
    make_mnist_forest(; rng=42, n_trees=100, max_depth=-1)

Construct a `RandomForestClassifier` suitable as a black-box
baseline on flattened MNIST features.

- `n_trees`: number of trees in the ensemble.
- `max_depth`: maximum depth of each tree (-1 means unlimited).
"""
function make_mnist_forest(; rng::Int = 42, n_trees::Int = 100, max_depth::Int = -1)
    model = RandomForestClassifier(n_trees = n_trees, max_depth = max_depth, rng = rng)
    return model
end

"""
    make_mnist_tree(; rng=42, max_depth=5)

Construct a single `DecisionTreeClassifier` for MNIST features.
Useful as a simpler black-box baseline.
"""
function make_mnist_tree(; rng::Int = 42, max_depth::Int = 5)
    model = DecisionTreeClassifier(max_depth = max_depth, rng = rng)
    return model
end

"""
    blackbox_predict(mach, X)

Pure prediction API for black-box attacks. Given an MLJ machine
and new features `X`, return probabilistic predictions.

This intentionally hides all training details and gradients.
"""
function blackbox_predict(mach, X)
    # For tree/forest models, `predict` already returns probabilistic predictions.
    return predict(mach, X)
end


"""
    SimpleConvBuilder

Minimal convolutional network builder for grayscale images such as MNIST.
This is intended as a white-box baseline for gradient-based attacks.
"""
struct SimpleConvBuilder
    filter_size::Int
    channels1::Int
    channels2::Int
end

# Helper for flattening to 2D inside the network
make2d(x::AbstractArray) = reshape(x, :, size(x)[end])

"""
    MLJFlux.build(b::SimpleConvBuilder, rng, n_in, n_out, n_channels)

Construct a small ConvNet for image classification.
"""
function MLJFlux.build(b::SimpleConvBuilder, rng, n_in, n_out, n_channels)
    k, c1, c2 = b.filter_size, b.channels1, b.channels2
    @assert isodd(k) "filter_size must be odd."

    p = div(k - 1, 2)
    init = Flux.glorot_uniform(rng)

    # assuming 28x28 input
    h, w = 28, 28
    h, w = div(h, 2), div(w, 2)   # after first MaxPool
    h, w = div(h, 2), div(w, 2)   # after second MaxPool
    flat_dim = h * w * c2

    model = Chain(
        Conv((k, k), n_channels => c1, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        x -> reshape(x, flat_dim, size(x, 4)),   # flatten to (flat_dim, batch)
        Dense(flat_dim, 128, relu),
        Dense(128, n_out),
        Flux.softmax,
    )

    return model
end

"""
    make_mnist_cnn(; rng=42, epochs=5, batch_size=64)

Create an `ImageClassifier` model suitable for MNIST-style
grayscale image classification.
"""
function make_mnist_cnn(; rng::Int = 42, epochs::Int = 5, batch_size::Int = 64)
    builder = SimpleConvBuilder(3, 16, 32)

    model = ImageClassifier(
        builder = builder,
        loss = Flux.Losses.crossentropy,
        optimiser = Optimisers.Adam(0.001),
        epochs = epochs,
        batch_size = batch_size,
        rng = rng,
    )

    return model
end

"""
    extract_flux_model(mach)

Return the underlying `Flux.Chain` from an MLJFlux `ImageClassifier`
machine, via MLJ's `fitted_params` API.
"""
function extract_flux_model(mach)
    fp = fitted_params(mach)
    # For MLJFlux ImageClassifier models, `fitted_params(mach)` exposes
    # the trained Flux network in the `chain` field.
    return fp.chain
end

"""
    SimpleConvBuilder

Minimal convolutional network builder for grayscale images such as MNIST.
This is intended as a white-box baseline for gradient-based attacks.
"""
struct SimpleConvBuilder
    filter_size::Int
    channels1::Int
    channels2::Int
end

# Helper for flattening to 2D inside the network
make2d(x::AbstractArray) = reshape(x, :, size(x)[end])

"""
    MLJFlux.build(b::SimpleConvBuilder, rng, n_in, n_out, n_channels)

Construct a small ConvNet for image classification.
"""
function MLJFlux.build(b::SimpleConvBuilder, rng, n_in, n_out, n_channels)
    k, c1, c2 = b.filter_size, b.channels1, b.channels2
    @assert isodd(k) "filter_size must be odd."

    p = div(k - 1, 2)
    init = Flux.glorot_uniform(rng)

    # assuming 28x28 input
    h, w = 28, 28
    h, w = div(h, 2), div(w, 2)   # after first MaxPool
    h, w = div(h, 2), div(w, 2)   # after second MaxPool
    flat_dim = h * w * c2

    model = Chain(
        Conv((k, k), n_channels => c1, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        x -> reshape(x, flat_dim, size(x, 4)),   # flatten to (flat_dim, batch)
        Dense(flat_dim, 128, relu),
        Dense(128, n_out),
        Flux.softmax,
    )

    return model
end

"""
    make_mnist_cnn(; rng=42, epochs=5, batch_size=64)

Create an `ImageClassifier` model suitable for MNIST-style
grayscale image classification.
"""
function make_mnist_cnn(; rng::Int = 42, epochs::Int = 5, batch_size::Int = 64)
    builder = SimpleConvBuilder(3, 16, 32)

    model = ImageClassifier(
        builder = builder,
        loss = Flux.Losses.crossentropy,
        optimiser = Optimisers.Adam(0.001),
        epochs = epochs,
        batch_size = batch_size,
        rng = rng,
    )

    return model
end

"""
    extract_flux_model(mach)

Return the underlying `Flux.Chain` from an MLJFlux `ImageClassifier`
machine, via MLJ's `fitted_params` API.
"""
function extract_flux_model(mach)
    fp = fitted_params(mach)
    # For MLJFlux ImageClassifier models, `fitted_params(mach)` exposes
    # the trained Flux network in the `chain` field.
    return fp.chain
end

end
