using Test
using AdversarialAttacks

const Model = AdversarialAttacks.FastGradientSignMethod.FGSM

@testset "FGSM Struct" begin

    # Test default constructor
    attack = Model()
    @test attack isa Model
    @test attack.parameters == Dict{String,Any}()
    
    # Test constructor with parameters
    params = Dict("epsilon" => 0.25)
    attack_with_params = Model(params)
    @test attack_with_params isa Model
    @test attack_with_params.parameters == params

    # Test type hierarchy
    @test Model <: AdversarialAttacks.WhiteBoxAttack
    @test Model <: AdversarialAttacks.AbstractAttack

    sample = [1.0, 2.0, 3.0]
    
    result = craft(sample, :m, attack_with_params)
    @test result == sample
    @test size(result) == size(sample)
    @test eltype(result) == eltype(sample)

end