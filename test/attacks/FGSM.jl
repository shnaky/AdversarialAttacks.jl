using Test
using AdversarialAttacks
using Flux

@testset "FGSM Struct" begin

    # Test default constructor
    atk = FGSM()
    @test atk isa FGSM
    @test atk.epsilon == 0.1

    # Test constructor with keyword argument
    atk_with_epsilon = FGSM(epsilon = 0.25)
    @test atk_with_epsilon isa FGSM
    @test atk_with_epsilon.epsilon == 0.25

    # Test type hierarchy
    @test FGSM <: WhiteBoxAttack
    @test FGSM <: AbstractAttack

    sample = (data = Float32[1.0, 2.0, 3.0], label = Flux.onehot(1, 1:2))
    model = Chain(
        Dense(3 => 2),
        softmax,
    )

    result = attack(atk_with_epsilon, model, sample)
    @test result isa Vector
    @test size(result) == size(sample.data)
    @test eltype(result) == eltype(sample.data)

end
