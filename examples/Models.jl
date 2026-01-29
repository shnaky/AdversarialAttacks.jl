module Models

using MLJ, MLJFlux, Flux, Optimisers, MLUtils
using NearestNeighborModels

export make_mnist_cnn, make_cifar_cnn
export make_forest, make_tree, make_knn, make_logistic, make_xgboost
export extract_flux_model
export SimpleConvBuilder

const DecisionTreeClassifier = @load DecisionTreeClassifier pkg = DecisionTree
const RandomForestClassifier = @load RandomForestClassifier pkg = DecisionTree
const KNNClassifier = @load KNNClassifier pkg = NearestNeighborModels
const LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels
const XGBoostClassifier = @load XGBoostClassifier pkg = XGBoost

const ImageClassifier = MLJFlux.ImageClassifier

# =========================
# Neural Network Models (White-box)
# =========================
struct SimpleConvBuilder
    filter_size::Int
    channels1::Int
    channels2::Int
    channels3::Int
end

"""
    MLJFlux.build(b::SimpleConvBuilder, rng, n_in, n_out, n_channels)

Build small ConvNet for image classification.
"""
function MLJFlux.build(b::SimpleConvBuilder, rng, n_in, n_out, n_channels)
    k, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3
    @assert isodd(k)
    p = div(k - 1, 2)
    init = Flux.glorot_uniform(rng)
    front = Chain(
        Conv((k, k), n_channels => c1, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        Conv((k, k), c2 => c3, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        MLUtils.flatten
    )
    d = Flux.outputsize(front, (n_in..., n_channels, 1)) |> first
    return Chain(front, Dense(d, n_out, init = init))
end

"""
    make_mnist_cnn(; rng=42, epochs=5, batch_size=64)

Create MLJFlux ImageClassifier for MNIST.
"""
function make_mnist_cnn(; rng::Int = 42, epochs::Int = 5, batch_size::Int = 64, kwargs...)
    builder = SimpleConvBuilder(3, 16, 32, 32)

    model = ImageClassifier(
        builder = builder,
        loss = Flux.Losses.crossentropy,
        optimiser = Optimisers.Adam(0.001),
        epochs = epochs,
        batch_size = batch_size,
        rng = rng,
        kwargs...
    )

    return model
end

"""
    extract_flux_model(mach)

Extract underlying Flux.Chain from MLJFlux machine.
"""
function extract_flux_model(mach)
    fp = fitted_params(mach)
    return fp.chain
end

"""
    make_cifar_cnn(; kwargs...)

CNN builder for CIFAR10 (3channels, 32x32) 
"""
function make_cifar_cnn(;
        epochs = 10,
        batch_size = 64,
        optimiser = Adam(),
        loss = Flux.Losses.crossentropy,
        kwargs...
    )
    builder = SimpleConvBuilder(3, 16, 32, 32)

    return ImageClassifier(
        builder = builder,
        epochs = epochs,
        batch_size = batch_size,
        optimiser = optimiser,
        loss = loss,
        kwargs...
    )
end


# =========================
# Traditional ML Models (Black-box)
# =========================
"""
    make_forest(; rng=42, n_trees=100, max_depth=-1)
"""
function make_forest(; rng::Int = 42, n_trees::Int = 100, max_depth::Int = -1, kwargs...)
    model = RandomForestClassifier(n_trees = n_trees, max_depth = max_depth, rng = rng, kwargs...)
    return model
end

"""
    make_tree(; rng=42, max_depth=5)

Construct a single `DecisionTreeClassifier` for MNIST features.
Useful as a simpler black-box baseline.
"""
function make_tree(; rng::Int = 42, max_depth::Int = 5, kwargs...)
    model = DecisionTreeClassifier(max_depth = max_depth, rng = rng, kwargs...)
    return model
end

"""
    make_knn(; K::Int = 5, weights = :uniform)

K-Nearest Neighbors classifier for MNIST.
"""
function make_knn(; K::Int = 5, weights::KNNKernel = Uniform(), kwargs...)
    model = KNNClassifier(K = K, weights = weights, kwargs...)
    return model
end

function make_logistic(; lambda::Float64 = 1.0e-4, penalty = :l2, kwargs...)
    model = LogisticClassifier(lambda = lambda, penalty = penalty, kwargs...)
    return model
end

"""
    make_xgboost(; num_round::Int = 100, max_depth::Int = 6, eta::Float64 = 0.3)

XGBoost classifier for MNIST.
"""
function make_xgboost(;
        num_round::Int = 100,
        max_depth::Int = 6,
        eta::Float64 = 0.3,
        rng::Int = 42,
        kwargs...
    )

    model = XGBoostClassifier(
        num_round = num_round,
        max_depth = max_depth,
        eta = eta,
        seed = rng,
        kwargs...
    )
    return model
end

end
