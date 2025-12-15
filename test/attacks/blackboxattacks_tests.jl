using Test
using AdversarialAttacks

const BRS_attack = AdversarialAttacks.BlackBoxAttacks.BasicRandomSearch

@testset "BasicRandomSearch Struct" begin

    # Test default constructor
    attack = BRS_attack()
    @test attack isa BRS_attack
    @test attack.parameters == Dict{String,Any}()
    
    # Test constructor with parameters
    params = Dict("epsilon" => 0.25)
    attack_with_params = BRS_attack(params)
    @test attack_with_params isa BRS_attack
    @test attack_with_params.parameters == params

    # Test type hierarchy
    @test BRS_attack <: AdversarialAttacks.BlackBoxAttack
    @test BRS_attack <: AdversarialAttacks.AbstractAttack

    sample = [1.0, 2.0, 3.0]
    
    result = craft(sample, :m, attack_with_params)
    @test result == sample
    @test size(result) == size(sample)
    @test eltype(result) == eltype(sample)

end

const square_attack = AdversarialAttacks.BlackBoxAttacks.SquareAttack

@testset "SquareAttack Struct" begin

    # Test default constructor
    attack = square_attack()
    @test attack isa square_attack
    @test attack.parameters == Dict{String,Any}()
    
    # Test constructor with parameters
    params = Dict("epsilon" => 0.25)
    attack_with_params = square_attack(params)
    @test attack_with_params isa square_attack
    @test attack_with_params.parameters == params

    # Test type hierarchy
    @test square_attack <: AdversarialAttacks.BlackBoxAttack
    @test square_attack <: AdversarialAttacks.AbstractAttack

    sample = [1.0, 2.0, 3.0]
    
    result = craft(sample, :m, attack_with_params)
    @test result == sample
    @test size(result) == size(sample)
    @test eltype(result) == eltype(sample)

end