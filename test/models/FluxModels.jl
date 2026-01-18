using Test
using AdversarialAttacks
using Flux

@testset "FluxModel wrapper" begin
    # Small toy network: 2 -> 2 -> 2
    m = Chain(Dense(2, 2, relu), Dense(2, 2))
    model = FluxModel(m)

    @test name(model) == "FluxModel"

    # Batch of 4 samples (2 features each)
    x = rand(Float32, 2, 4)
    # One-hot labels for 4-class toy problem
    y = Flux.onehotbatch([1, 2, 1, 2], 1:2)

    ŷ = predict(model, x)
    @test size(ŷ) == size(y)          # output shape matches labels

    ℓ = loss(model, x, y)
    @test ℓ isa Real                  # loss is a scalar

    θ = params(model)
    @test length(θ) > 0               # has trainable parameters
end

@testset "Pretrained CIFAR-10 model" begin
    model = load_pretrained_c10_model()

    @test name(model) == "FluxModel"
    @test model isa FluxModel
    @test model.model isa Flux.Chain

    # Batch of 8 CIFAR-10 images (3x32x32)
    x = rand(Float32, 32, 32, 3, 8)
    # One-hot labels for 10-class CIFAR-10
    y = Flux.onehotbatch(rand(1:10, 8), 1:10)

    ŷ = predict(model, x)
    @test size(ŷ) == size(y)          # output shape matches labels

    ℓ = loss(model, x, y)
    @test ℓ isa Real                  # loss is a scalar

    θ = params(model)
    @test length(θ) > 0               # has trainable parameters
end
