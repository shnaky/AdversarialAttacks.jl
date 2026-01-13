using Test
using AdversarialAttacks

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

    # Minimal dummy model with MSE loss
    struct TestModel <: Model.AbstractModel end
    Model.loss(::TestModel, x, y) = sum((x .- y).^2)

    # Create sample with data and label fields
    sample = (data=[1.0, 2.0, 3.0], label=[0.0, 1.0, 0.0])
    model = TestModel()
    
    result = craft(sample, model, attack_with_params)
    @test size(result) == size(sample.data)
    @test eltype(result) == eltype(sample.data)

end