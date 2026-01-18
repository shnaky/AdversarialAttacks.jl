using Test
using AdversarialAttacks
using Flux
using Flux: Chain, Dense

@testset "FGSM Struct" begin

    # Test default constructor
    attack = FGSM()
    @test attack isa FGSM
    @test attack.epsilon == 0.1

    # Test constructor with keyword argument
    attack_with_epsilon = FGSM(epsilon=0.25)
    @test attack_with_epsilon isa FGSM
    @test attack_with_epsilon.epsilon == 0.25
    @test hyperparameters(attack_with_epsilon) == Dict("epsilon" => 0.25)

    # Test type hierarchy
    @test FGSM <: WhiteBoxAttack
    @test FGSM <: AbstractAttack

    # Test with minimal DifferentiableModel (teammate's test)
    struct TestModel <: DifferentiableModel end
    AdversarialAttacks.loss(::TestModel, x, y) = sum((x .- y).^2)

    sample_simple = (data=[1.0, 2.0, 3.0], label=[0.0, 1.0, 0.0])
    test_model = TestModel()

    result_simple = craft(sample_simple, test_model, attack_with_epsilon)
    @test size(result_simple) == size(sample_simple.data)
    @test eltype(result_simple) == eltype(sample_simple.data)

    # Test with NamedTuple sample format and proper FluxModel
    sample = (data=Float32[1.0, 2.0, 3.0], label=Flux.onehot(1, 1:2))
    model = FluxModel(Chain(Dense(3 => 2)))

    result = craft(sample, model, attack_with_epsilon)
    @test result isa Vector
    @test size(result) == size(sample.data)
    @test eltype(result) == eltype(sample.data)

end
