module Experiments

include("Models.jl")
using .Models

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
using ScientificTypes: ColorImage

using BSON
using Dates

export ExperimentConfig, run_experiment
export load_mnist_for_mlj, flatten_images, load_cifar10_for_mlj
export make_mnist_cnn, make_cifar_cnn
export make_mnist_forest, make_mnist_tree, make_mnist_knn, make_mnist_logistic, make_mnist_xgboost
export blackbox_predict, extract_flux_model
export save_experiment_result, load_experiment_result, get_or_train

const MODELS_DIR = joinpath(@__DIR__, ".", "models")

# =========================
# Config & Data
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

Returns:
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

"""
    load_cifar10_for_mlj(; n_train::Int = 50000)

MLJFlux ColorImage for CIFAR10. C×H×W, Float32 Array{3}.
"""
function load_cifar10_for_mlj(; n_train::Int = 50000)
    dataset = MLDatasets.CIFAR10(split = :train)

    # raw 4D array (32x32x3xN Float32 HWC)
    images = dataset.features[:, :, :, 1:n_train]

    images = coerce(images, ColorImage)

    labels = coerce(dataset.targets[1:n_train], Multiclass{10})

    return images, labels  # Array{ColorImage,4}
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

# =========================
# Model Persistence (Save/Load)
# =========================

"""
    save_experiment_result(result, name::String)

Save trained experiment result to disk.

# Arguments
- `result`: Output from run_experiment()
- `name`: Model name (e.g., "mnist_cnn_comparison")

# Example
```julia
result = run_experiment(cnn_model, X, y)
save_experiment_result(result, "my_cnn")
```

"""
function save_experiment_result(result, name::String)
    mkpath(MODELS_DIR)
    model_path = joinpath(MODELS_DIR, "$(name).jlso")
    meta_path = joinpath(MODELS_DIR, "$(name)_meta.bson")

    MLJ.save(model_path, result.mach)

    metadata = Dict(
        "test_idx" => result.test_idx,
        "y_test" => result.y_test,
        "accuracy" => result.report.accuracy,
        "trained_at" => now(),
    )
    BSON.@save meta_path metadata

    println("✓ Model saved:")
    println("  • Machine: $model_path")
    println("  • Metadata: $meta_path")
    println("  • Accuracy: ", round(metadata["accuracy"] * 100, digits = 2), "%")

    return (model_path, meta_path)
end

"""
load_experiment_result(name::String)

Load previously saved experiment result.

# Arguments

- `name`: Model name (same as in save_experiment_result)


# Returns

- `(mach, metadata)` if found, `nothing` otherwise


# Example

```julia
mach, meta = load_experiment_result("my_cnn")
flux_model = extract_flux_model(mach)
```

"""
function load_experiment_result(name::String)
    model_path = joinpath(MODELS_DIR, "$(name).jlso")
    meta_path = joinpath(MODELS_DIR, "$(name)_meta.bson")

    if !isfile(model_path) || !isfile(meta_path)
        @warn "Model not found: $name"
        return nothing
    end

    println("📦 Loading saved model: $name")
    mach = MLJ.machine(model_path)
    meta = BSON.load(meta_path)[:metadata]

    println("✓ Model loaded:")
    println("  • Accuracy: ", round(meta["accuracy"] * 100, digits = 2), "%")
    println("  • Trained: ", meta["trained_at"])

    return (mach, meta)
end

"""
    get_or_train(model_factory::Function, name::String; 
                 force_retrain=false, use_flatten::Bool=true, kwargs...)

Generic trainer for ANY MLJ model.

- `model_factory`: make_mnist_* function
- `use_flatten`: image → DataFrame (Tree/KNN/..etc=yes, CNN=no)
- `kwargs...`: Additional arguments for training (e.g., epochs, batch_size)

# Example

```julia
mach, meta = get_or_train_cnn("comparison_cnn", epochs=10, batch_size=64)
flux_model = extract_flux_model(mach)
```

"""
function get_or_train(
        model_factory::Function, name::String;
        config::ExperimentConfig = DEFAULT_CONFIG,
        dataset = :mnist,  # :mnist or :cifar10
        force_retrain = false, use_flatten = true, kwargs...
    )

    if !force_retrain
        cached = load_experiment_result(name)
        if !isnothing(cached)
            println("📦 Loaded cached $name (Acc: $(round(cached[2]["accuracy"] * 100, digits = 1))%)")
            return cached
        end
    end

    println("🚀 Training $name on $dataset...")
    if dataset == :mnist
        X_img, y = load_mnist_for_mlj()
    elseif dataset == :cifar10
        X_img, y = load_cifar10_for_mlj()
    else
        error("Unsupported dataset: $dataset. Use :mnist or :cifar10.")
    end
    X = use_flatten ? flatten_images(X_img) : X_img

    model = model_factory(; kwargs...)
    result = run_experiment(model, X, y; config = config)

    save_experiment_result(result, name)
    meta = Dict(
        "test_idx" => result.test_idx,
        "y_test" => result.y_test,
        "accuracy" => result.report.accuracy,
        "trained_at" => now(),
        "model_type" => nameof(typeof(model))
    )

    println("✅ $name complete: $(round(meta["accuracy"] * 100, digits = 1))%")
    return (result.mach, meta)
end

end
