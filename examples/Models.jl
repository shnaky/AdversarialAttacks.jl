module Models

using MLJFlux, Flux, Optimisers

using MLJ
using NearestNeighborModels

export make_mnist_cnn, make_cifar_cnn
export make_mnist_forest, make_mnist_tree, make_mnist_knn, make_mnist_logistic, make_mnist_xgboost
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
    h::Int
    w::Int
end

"""
    FlattenLayer

Named flatten layer for reliable serialization.
Reshapes 4D image tensor (H, W, C, N) to 2D matrix (flat_dim, N).
"""
struct FlattenLayer end

function (f::FlattenLayer)(x)
    return reshape(x, :, size(x, ndims(x)))
end

"""
    MLJFlux.build(b::SimpleConvBuilder, rng, n_in, n_out, n_channels)

Build small ConvNet for image classification.
Assumes 28×28 input images.
"""
function MLJFlux.build(b::SimpleConvBuilder, rng, n_in, n_out, n_channels)
    k, c1, c2 = b.filter_size, b.channels1, b.channels2
    @assert isodd(k)
    p = div(k - 1, 2)
    init = Flux.glorot_uniform(rng)
    h = b.h; w = b.w; h, w = div(h, 2), div(w, 2); h, w = div(h, 2), div(w, 2)
    flat_dim = h * w * c2
    return Chain(
        Conv((k, k), n_channels => c1, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad = (p, p), relu, init = init),
        MaxPool((2, 2)),
        FlattenLayer(),
        Dense(flat_dim, 128, relu, init = init),
        Dense(128, n_out, init = init)
    )
end

"""
    make_mnist_cnn(; rng=42, epochs=5, batch_size=64)

Create MLJFlux ImageClassifier for MNIST.
"""
function make_mnist_cnn(; rng::Int = 42, epochs::Int = 5, batch_size::Int = 64)
    builder = SimpleConvBuilder(3, 16, 32, 28, 28)

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
    builder = SimpleConvBuilder(3, 16, 32, 32, 32)

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
    make_mnist_forest(; rng=42, n_trees=100, max_depth=-1)
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
    make_mnist_knn(; K::Int = 5, weights = :uniform)

K-Nearest Neighbors classifier for MNIST.
"""
function make_mnist_knn(; K::Int = 5, weights::KNNKernel = Uniform())
    model = KNNClassifier(K = K, weights = weights)
    return model
end

function make_mnist_logistic(; lambda::Float64 = 1.0e-4, penalty = :l2)
    model = LogisticClassifier(lambda = lambda, penalty = penalty)
    return model
end

"""
    make_mnist_xgboost(; num_round::Int = 100, max_depth::Int = 6, eta::Float64 = 0.3)

XGBoost classifier for MNIST.
"""
function make_mnist_xgboost(;
        num_round::Int = 100,
        max_depth::Int = 6,
        eta::Float64 = 0.3,
        rng::Int = 42
    )

    model = XGBoostClassifier(
        num_round = num_round,
        max_depth = max_depth,
        eta = eta,
        seed = rng
    )
    return model
end

end
