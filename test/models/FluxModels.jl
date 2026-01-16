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
