using Test
using AdversarialAttacks

const FGSM_attack = AdversarialAttacks.FastGradientSignMethod.FGSM

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

    sample = [1.0, 2.0, 3.0]
    
    result = craft(sample, :m, attack_with_params)
    @test result == sample
    @test size(result) == size(sample)
    @test eltype(result) == eltype(sample)

end