using Test
using AdversarialAttacks
using Flux
using Flux: Chain, Dense

const FGSM_attack = AdversarialAttacks.FastGradientSignMethod.FGSM
const Model = AdversarialAttacks.Model

@testset "FGSM Struct" begin

    # Test default constructor
    attack = FGSM_attack()
    @test attack isa FGSM_attack
    @test attack.parameters == Dict{String,Any}()

    # Test constructor with parameters
    params = Dict("epsilon" => 0.25)
    attack_with_params = FGSM_attack(params)
    @test attack_with_params isa FGSM_attack
    @test hyperparameters(attack_with_params) == params

    # Test type hierarchy
    @test FGSM_attack <: AdversarialAttacks.WhiteBoxAttack
    @test FGSM_attack <: AdversarialAttacks.AbstractAttack

    # Test with minimal DifferentiableModel (teammate's test)
    struct TestModel <: Model.DifferentiableModel end
    Model.loss(::TestModel, x, y) = sum((x .- y).^2)

    sample_simple = (data=[1.0, 2.0, 3.0], label=[0.0, 1.0, 0.0])
    test_model = TestModel()

    result_simple = craft(sample_simple, test_model, attack_with_params)
    @test size(result_simple) == size(sample_simple.data)
    @test eltype(result_simple) == eltype(sample_simple.data)

    # Test with NamedTuple sample format and proper FluxModel
    sample = (data=Float32[1.0, 2.0, 3.0], label=Flux.onehot(1, 1:2))
    model = FluxModel(Chain(Dense(3 => 2)))

    result = craft(sample, model, attack_with_params)
    @test result isa Vector
    @test size(result) == size(sample.data)
    @test eltype(result) == eltype(sample.data)

end
