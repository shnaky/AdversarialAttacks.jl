using Test
using AdversarialAttacks
using Flux

@testset "FGSM Struct" begin

    # Test default constructor
    attack = FGSM()
    @test attack isa FGSM
    @test attack.epsilon == 0.1

    # Test constructor with keyword argument
    attack_with_epsilon = FGSM(epsilon = 0.25)
    @test attack_with_epsilon isa FGSM
    @test attack_with_epsilon.epsilon == 0.25
    @test hyperparameters(attack_with_epsilon) == Dict("epsilon" => 0.25)

    # Test type hierarchy
    @test FGSM <: WhiteBoxAttack
    @test FGSM <: AbstractAttack

    sample = (data = Float32[1.0, 2.0, 3.0], label = Flux.onehot(1, 1:2))
    model = Chain(
        Dense(3 => 2),
        softmax,
    )

    result = craft(sample, model, attack_with_epsilon)
    @test result isa Vector
    @test size(result) == size(sample.data)
    @test eltype(result) == eltype(sample.data)

end
