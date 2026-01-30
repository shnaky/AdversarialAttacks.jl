using MLJ: @load, fitted_params
import MLJFlux: build # build must be explicitly imported
using MLJFlux: ImageClassifier
using Flux: glorot_uniform, outputsize, Conv, Chain, Dense, MaxPool, relu, crossentropy, BatchNorm, Dropout
using Optimisers: Adam
using NearestNeighborModels: KNNKernel
using MLUtils: flatten

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
export make_mnist_cnn, make_cifar_cnn
export make_forest, make_tree, make_knn, make_logistic, make_xgboost
export extract_flux_model

# ------------------------------------------------------------------
# Aliases for MLJ models
# ------------------------------------------------------------------
const DecisionTreeClassifier = @load DecisionTreeClassifier pkg = DecisionTree
const RandomForestClassifier = @load RandomForestClassifier pkg = DecisionTree
const KNNClassifier = @load KNNClassifier pkg = NearestNeighborModels
const LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels
const XGBoostClassifier = @load XGBoostClassifier pkg = XGBoost

# ------------------------------------------------------------------
# Convolutional builders (white-box models)
# ------------------------------------------------------------------

"""
    SimpleConvBuilder

Lightweight configuration object for constructing a small ConvNet
suitable for MNIST or CIFAR-10 through `MLJFlux.ImageClassifier`.

The actual network is defined in `MLJFlux.build(::SimpleConvBuilder, ...)`.
"""
struct SimpleConvBuilder
    filter_size::Int
    channels1::Int
    channels2::Int
    channels3::Int
end
SimpleConvBuilder() = SimpleConvBuilder(3, 16, 32, 32)

"""
    MLJFlux.build(b::SimpleConvBuilder, rng, n_in, n_out, n_channels)

Builds a simple convolutional classifier for image inputs.

- `n_in`      : tuple `(height, width)`
- `n_out`     : number of classes
- `n_channels`: number of input channels (1 for gray, 3 for RGB)

The resulting architecture is:

Conv -> MaxPool -> Conv -> MaxPool -> Conv -> MaxPool -> Flatten -> Dense
"""
function build(b::SimpleConvBuilder, rng, n_in, n_out, n_channels)
    k, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3
    @assert isodd(k) "filter_size must be odd to keep spatial dimensions aligned"

    p = div(k - 1, 2)
    init = glorot_uniform(rng)

    front = Chain(
        Conv((k, k), n_channels => c1, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        Conv((k, k), c2 => c3, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        flatten,
    )

    d = first(outputsize(front, (n_in..., n_channels, 1)))

    return Chain(front, Dense(d, n_out, init = init))
end

"""
    CifarConvBuilder

AI-assisted configuration for CIFAR-10 ConvNet (slightly tuned channels/depth).

Construct via `CifarConvBuilder(3, 32, 64, 128, 0.25f0)`.
"""
struct CifarConvBuilder
    filter_size::Int
    channels1::Int
    channels2::Int
    channels3::Int
    dropout::Float32
end
CifarConvBuilder() = CifarConvBuilder(3, 32, 64, 128, 0.25f0)

"""
    MLJFlux.build(b::CifarConvBuilder, rng, n_in, n_out, n_channels)

Builds CIFAR-10 optimized convolutional classifier for 32×32×3 RGB images.

Architecture: Conv-BN-MP ×3 → Flatten → Dropout → Dense(256) → Dropout → Dense(n_out)

- `n_in`: `(32, 32)`
- `n_channels`: `3` (RGB)
- BatchNorm/Dropout for regularization and training stability

Returns `Flux.Chain` compatible with `MLJFlux.ImageClassifier`.
"""
function build(b::CifarConvBuilder, rng, n_in, n_out, n_channels)
    k, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3

    p = div(k - 1, 2)
    init = glorot_uniform(rng)

    conv_layers = Chain(
        Conv((k, k), n_channels => c1, pad = (p, p), relu, init = init),
        BatchNorm(c1),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad = (p, p), relu, init = init),
        BatchNorm(c2),
        MaxPool((2, 2)),
        Conv((k, k), c2 => c3, pad = (p, p), relu, init = init),
        BatchNorm(c3),
        MaxPool((2, 2)),
        flatten,
    )
    d_conv = first(outputsize(conv_layers, (n_in..., n_channels, 1)))

    front = Chain(
        conv_layers,
        Dropout(b.dropout),
        Dense(d_conv, 256, relu),
        Dropout(b.dropout)
    )

    return Chain(front, Dense(256, n_out, init = init))
end

# ------------------------------------------------------------------
# CNN wrappers for MLJFlux
# ------------------------------------------------------------------

"""
    make_mnist_cnn(; rng=42, epochs=5, batch_size=64, kwargs...)

Construct an `MLJFlux.ImageClassifier` configured for MNIST-like data
(single-channel 28×28 images) using a small ConvNet.

Additional keyword arguments are forwarded to `ImageClassifier`.
"""
function make_mnist_cnn(; rng::Int = 42, epochs::Int = 5, batch_size::Int = 64, loss = crossentropy, optimiser = Adam(0.001), lambda = 1.0e-4, kwargs...)
    builder = SimpleConvBuilder(3, 16, 32, 32)

    return ImageClassifier(
        builder = builder,
        loss = loss,
        optimiser = optimiser,
        epochs = epochs,
        batch_size = batch_size,
        rng = rng,
        lambda = lambda,
        kwargs...,
    )
end


"""
    make_cifar_cnn(; epochs=50, batch_size=128, optimiser=Adam(), loss=crossentropy, lambda = 5.0e-4, kwargs...)

Construct an `MLJFlux.ImageClassifier` for CIFAR-10-like data
(3-channel 32×32 images) using the same ConvNet recipe as for MNIST,
but with typically more training epochs.

All hyperparameters can be overridden via keywords.
"""
function make_cifar_cnn(;
        epochs::Int = 50,
        batch_size::Int = 128,
        optimiser = Adam(0.001),
        loss = crossentropy,
        lambda = 5.0e-4,
        kwargs...,
    )
    builder = CifarConvBuilder(3, 32, 64, 128, 0.25)

    return ImageClassifier(
        builder = builder,
        epochs = epochs,
        batch_size = batch_size,
        optimiser = optimiser,
        lambda = lambda,
        loss = loss,
        kwargs...,
    )
end

"""
    extract_flux_model(mach) -> Flux.Chain

Extract the underlying `Flux.Chain` from a fitted MLJFlux `machine`.

Note: The returned chain does **not** include any MLJ preprocessing
such as coercions or rescaling; it corresponds only to the raw
neural network part used by `ImageClassifier`.
"""
function extract_flux_model(mach)
    fp = fitted_params(mach)
    return fp.chain
end

# ------------------------------------------------------------------
# Classical ML models (black-box baselines)
# ------------------------------------------------------------------

"""
    make_forest(; rng=42, n_trees=100, max_depth=-1, kwargs...) -> RandomForestClassifier

Generic random forest classifier factory.

Suitable for both MNIST and CIFAR-10 once the images are flattened
into tabular features. Additional keyword arguments are forwarded to
`RandomForestClassifier`.
"""
function make_forest(;
        rng::Int = 42,
        n_trees::Int = 100,
        max_depth::Int = -1,
        kwargs...,
    )
    return RandomForestClassifier(
        n_trees = n_trees,
        max_depth = max_depth,
        rng = rng,
        kwargs...,
    )
end

"""
    make_tree(; rng=42, max_depth=5, kwargs...) -> DecisionTreeClassifier

Single CART decision tree baseline. Works on any tabular representation
of MNIST or CIFAR-10 features.
"""
function make_tree(; rng::Int = 42, max_depth::Int = 5, kwargs...)
    return DecisionTreeClassifier(
        max_depth = max_depth,
        rng = rng,
        kwargs...,
    )
end

"""
    make_knn(; K=5, weights=Uniform(), kwargs...) -> KNNClassifier

k-Nearest Neighbors classifier for flattened image features.

- `K`       : number of neighbors
- `weights` : instance of `KNNKernel` (e.g. `Uniform()`, `DistanceWeighted()`)

Additional keyword arguments are forwarded to `KNNClassifier`.
"""
function make_knn(; K::Int = 5, weights::KNNKernel = Uniform(), kwargs...)
    return KNNClassifier(
        K = K,
        weights = weights,
        kwargs...,
    )
end

"""
    make_logistic(; lambda=1e-4, penalty=:l2, kwargs...) -> LogisticClassifier

Multinomial logistic regression baseline on tabular features.

- `lambda` : ℓ2/ℓ1 regularization strength
- `penalty`: `:l2` or `:l1`

Additional keyword arguments are forwarded to `LogisticClassifier`.
"""
function make_logistic(; lambda::Float64 = 1.0e-4, penalty = :l2, kwargs...)
    return LogisticClassifier(
        lambda = lambda,
        penalty = penalty,
        kwargs...,
    )
end

"""
    make_xgboost(; num_round=100, max_depth=6, eta=0.3, rng=42, kwargs...) -> XGBoostClassifier

Generic XGBoost classifier factory for tabular features.

This is dataset-agnostic and can be used for MNIST or CIFAR-10
once images are flattened. Hyperparameters mirror common XGBoost
naming:

- `num_round` : number of boosting iterations
- `max_depth` : depth of each tree
- `eta`       : learning rate
- `rng`       : random seed

Additional keyword arguments are forwarded to `XGBoostClassifier`.
"""
function make_xgboost(;
        num_round::Int = 100,
        max_depth::Int = 6,
        eta::Float64 = 0.3,
        rng::Int = 42,
        kwargs...,
    )
    return XGBoostClassifier(
        num_round = num_round,
        max_depth = max_depth,
        eta = eta,
        seed = rng,
        kwargs...,
    )
end
