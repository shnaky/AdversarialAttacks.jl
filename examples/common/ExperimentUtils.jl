module ExperimentUtils

include("Models.jl")

using MLJ
using MLJ: partition, accuracy
using MLJFlux
using Flux
using Optimisers

using MLDatasets
using DataFrames
using StatsBase: mode

using ColorTypes: Color, Gray, RGB
using Images: channelview
using ScientificTypes: ColorImage

using BSON
using Dates

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
export ExperimentConfig, run_experiment
export make_mnist_cnn, make_cifar_cnn, extract_flux_model
export make_forest, make_tree, make_knn, make_logistic, make_xgboost
export save_experiment_result, load_experiment_result, get_or_train
export DatasetType, DATASET_MNIST, DATASET_CIFAR10, load_data, dataset_shapes

const MODELS_DIR = joinpath(@__DIR__, "..", "models")

# ------------------------------------------------------------------
# Configuration and dataset metadata
# ------------------------------------------------------------------

"""
    DatasetType

Enum flag indicating which dataset an experiment uses.
Currently supports:

- `DATASET_MNIST`
- `DATASET_CIFAR10`
"""
@enum DatasetType DATASET_MNIST DATASET_CIFAR10

"""
    ExperimentConfig

Lightweight configuration struct describing a single experiment:

- `exp_name`       : human-readable experiment name
- `model_file_name`: base name for persisted model/metadata files
- `model_factory`  : function producing an MLJ model (e.g. `make_mnist_cnn`)
- `dataset`        : `DatasetType` enum
- `use_flatten`    : if `true`, images are flattened to tabular features
- `force_retrain`  : retrain even if a cached model exists
- `fraction_train`    : train fraction for the train/test split
- `rng`            : integer seed for reproducible splits and training
- `model_hyperparams`: named tuple of kwargs forwarded to `model_factory`
"""
Base.@kwdef struct ExperimentConfig
    exp_name::String = "default_exp"
    model_file_name::String = "mnist_cnn"
    model_factory::Function = make_mnist_cnn
    dataset::DatasetType = DATASET_MNIST
    use_flatten::Bool = false
    force_retrain::Bool = false
    fraction_train::Float64 = 0.8
    rng::Int = 42
    model_hyperparams::NamedTuple = NamedTuple()
end

"""
    dataset_shapes

Map from `DatasetType` to `(height, width, channels)` used when
reshaping images into Flux tensors.
"""
dataset_shapes = Dict(
    DATASET_MNIST => (28, 28, 1),
    DATASET_CIFAR10 => (32, 32, 3),
)

# ------------------------------------------------------------------
# Data loading utilities
# ------------------------------------------------------------------

"""
    flatten_images(X_img::Vector{<:AbstractMatrix{<:Gray}})

Convert a vector of H×W×C image arrays into a `DataFrame` suitable for
tabular MLJ models (trees, linear models, etc.). Each pixel becomes one
feature column (e.g. `x1, x2, ...`).
"""
function flatten_images(X_img::Vector{<:AbstractMatrix{<:Gray}})
    n = length(X_img)
    d = length(vec(X_img[1]))

    Xmat = Array{Float32}(undef, n, d)
    for i in 1:n
        Xmat[i, :] .= vec(X_img[i])
    end

    return DataFrame(Xmat, :auto)
end

"""
    flatten_images(X_img::Vector{<:AbstractMatrix{<:RGB}})

Flatten a vector of `RGB` images into a `DataFrame` with one row per
image and 3×32×32 columns (channel-first after `channelview`).
"""
function flatten_images(X_img::Vector{<:AbstractMatrix{<:RGB}})
    h, w, c = dataset_shapes[DATASET_CIFAR10]
    n = length(X_img)
    d = h * w * c

    Xmat = Array{Float32}(undef, n, d)
    for i in 1:n
        Xmat[i, :] .= vec(channelview(X_img[i]))
    end

    return DataFrame(Xmat, :auto)
end

"""
    load_dataset_for_mlj(::Val{DATASET_MNIST}; n_train::Int = 60000)

Load the MNIST training split and return `(X, y)` in a form compatible
with `MLJFlux.ImageClassifier`.

Returns:

- `X`: `Vector` of 28×28 `Gray` images (values in [0, 1])
- `y`: `CategoricalVector` of digit labels (0–9) with `Multiclass` scitype
"""
function load_dataset_for_mlj(::Val{DATASET_MNIST}; n_train::Int = 60000)
    # images: 28×28×N UInt8, labels: Vector{Int}
    images, labels = MLDatasets.MNIST(split = :train)[:]
    images = images[:, :, 1:n_train]
    labels = labels[1:n_train]

    # Convert to Gray images in [0, 1]
    X_vec = [Gray.(images[:, :, i] ./ 255) for i in 1:n_train]

    # Labels must be categorical for MLJ
    y = coerce(labels, Multiclass)

    return X_vec, y
end

"""
    load_dataset_for_mlj(::Val{DATASET_CIFAR10}; n_train::Int = 50000)

Load the CIFAR-10 training split and return `(X, y)` in a form
compatible with `MLJFlux.ImageClassifier`.

Returns:

- `X`: 4D array of `ColorImage` (as required by MLJFlux)
- `y`: `CategoricalVector` with `Multiclass{10}` scitype
"""
function load_dataset_for_mlj(::Val{DATASET_CIFAR10}; n_train::Int = 50000)
    dataset = MLDatasets.CIFAR10(split = :train)

    # raw 4D array: 32×32×3×N (HWC layout)
    images = dataset.features[:, :, :, 1:n_train]

    # Coerce to `ColorImage` so MLJFlux recognises the images correctly
    images = coerce(images, ColorImage)

    # Targets to categorical Multiclass{10}
    labels = coerce(dataset.targets[1:n_train], Multiclass{10})

    return images, labels
end

# ------------------------------------------------------------------
# Dataset loading dispatcher
# ------------------------------------------------------------------

"""
    load_data(dataset::DatasetType, use_flatten)

Load `(X, y)` for the given `dataset` and optionally flatten images
to tabular features if `use_flatten == true`.

- For `DATASET_MNIST`:
  - `use_flatten = false`: vector of `Gray` images
  - `use_flatten = true` : tabular `DataFrame` of pixel features

- For `DATASET_CIFAR10`:
  - `use_flatten = false`: 4D `ColorImage` array
  - `use_flatten = true` : tabular `DataFrame` of pixel features
"""
function load_data(dataset::DatasetType, use_flatten::Bool)
    X_img, y = load_dataset_for_mlj(Val(dataset))

    X = use_flatten ? flatten_images(X_img) : X_img

    return X, y
end

# ------------------------------------------------------------------
# Train/test splitting and baseline experiment
# ------------------------------------------------------------------

"""
    train_test_split(n; fraction_train, rng)

Partition indices `1:n` into `(train, test)` using MLJ's `partition`.

- `fraction_train`: fraction of observations for the training set
- `rng`           : integer seed for reproducible shuffling
"""
function train_test_split(n::Integer; fraction_train::Float64, rng::Int)
    train, test = partition(1:n, fraction_train, shuffle = true, rng = rng)
    return train, test
end

"""
    run_experiment(model, X, y; config)

Fit the given MLJ `model` on `(X, y)` using a simple train/test split
as specified in `config`.

Returns a named tuple with:

- `mach`     : fitted MLJ machine
- `train_idx`: training indices
- `test_idx` : test indices
- `y_pred_test`  : probabilistic predictions on the test set
- `y_test`   : true labels on the test set
- `config`   : the `ExperimentConfig` used
- `report`   : small summary named tuple (currently only `accuracy`)

This function does *not* perform any adversarial attacks; it only
sets up a clean baseline experiment.
"""
function run_experiment(model, X, y; config::ExperimentConfig)
    n = length(y)
    train, test = train_test_split(n; fraction_train = config.fraction_train, rng = config.rng)

    # For DataFrame inputs we must use two-dimensional indexing
    Xtrain = X isa DataFrame ? X[train, :] : X[train]
    Xtest = X isa DataFrame ? X[test, :] : X[test]

    mach = machine(model, Xtrain, y[train])
    fit!(mach, verbosity = 1)

    # Probabilistic predictions on the test set
    y_pred_test = predict(mach, Xtest)

    # Convert to point predictions via `mode`
    acc = accuracy(mode.(y_pred_test), y[test])

    report = (accuracy = acc,)

    return (
        mach = mach,
        train_idx = train,
        test_idx = test,
        y_pred_test = y_pred_test,
        y_test = y[test],
        config = config,
        report = report,
    )
end

# ------------------------------------------------------------------
# Model persistence (save / load)
# ------------------------------------------------------------------

"""
    save_experiment_result(result, name)

Persist a trained experiment result to disk.

- Saves the MLJ `machine` as `models/<name>.jlso`
- Saves metadata (indices, accuracy, timestamp) as `models/<name>_meta.bson`

Returns `(model_path, meta_path)`.
"""
function save_experiment_result(result, name::String)
    mkpath(MODELS_DIR)
    model_path = joinpath(MODELS_DIR, "$(name).jlso")
    meta_path = joinpath(MODELS_DIR, "$(name)_meta.bson")

    # Save the trained machine (uses JLD2 under the hood)
    MLJ.save(model_path, result.mach)

    metadata = Dict(
        "test_idx" => result.test_idx,
        "y_test" => result.y_test,
        "accuracy" => result.report.accuracy,
        "trained_at" => now(),
    )
    BSON.@save meta_path metadata

    println("✓ Model saved:")
    println("  • Machine:  $model_path")
    println("  • Metadata: $meta_path")
    println("  • Accuracy: ", round(metadata["accuracy"] * 100, digits = 2), "%")

    return (model_path, meta_path)
end

"""
    load_experiment_result(name) -> (mach, meta) or `nothing`

Load a previously saved experiment by name.

- Expects `models/<name>.jlso` and `models/<name>_meta.bson`.
- Returns `(mach, metadata::Dict)` if both files exist,
  otherwise returns `nothing`.
"""
function load_experiment_result(name::String)
    model_path = joinpath(MODELS_DIR, "$(name).jlso")
    meta_path = joinpath(MODELS_DIR, "$(name)_meta.bson")

    if !isfile(model_path) || !isfile(meta_path)
        @warn "Model not found: $name"
        return nothing
    end

    println("📦 Loading saved model: $name")

    # Reconstruct the MLJ machine from the serialized object
    mach = MLJ.machine(model_path)
    meta = BSON.load(meta_path)[:metadata]

    println("✓ Model loaded:")
    println("  • Accuracy: ", round(meta["accuracy"] * 100, digits = 2), "%")
    println("  • Trained:  ", meta["trained_at"])

    return (mach, meta)
end

# ------------------------------------------------------------------
# High-level training helper
# ------------------------------------------------------------------

"""
    get_or_train(config::ExperimentConfig)

High-level helper to either load a cached trained model (if available)
or train a new one according to `config`.

- If `config.force_retrain` is `false`, attempts to load
  `<config.model_file_name>` from disk.
- Otherwise (or on cache miss), loads data, instantiates the model via
  `config.model_factory`, runs `run_experiment`, and saves the result.

Returns `(mach, meta)` where `meta` is a `Dict` matching what
`load_experiment_result` would return.
"""
function get_or_train(config::ExperimentConfig)
    if !config.force_retrain
        cached = load_experiment_result(config.model_file_name)
        if !isnothing(cached)
            acc = round(cached[2]["accuracy"] * 100, digits = 1)
            println("📦 Loaded cached $(config.model_file_name) (Acc: $(acc)%)")
            return cached
        end
    end

    println("🚀 Training $(config.model_file_name) on $(config.dataset)...")

    X, y = load_data(config.dataset, config.use_flatten)

    model = config.model_factory(; config.model_hyperparams...)
    result = run_experiment(model, X, y; config = config)

    save_experiment_result(result, config.model_file_name)

    meta = Dict(
        "test_idx" => result.test_idx,
        "y_test" => result.y_test,
        "accuracy" => result.report.accuracy,
        "trained_at" => now(),
        "model_type" => nameof(typeof(model)),
    )

    println(
        "✅ $(config.model_file_name) complete: ",
        round(meta["accuracy"] * 100, digits = 1), "%"
    )

    return (result.mach, meta)
end

end
