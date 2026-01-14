using Test
using AdversarialAttacks

# Shared dummy model for black-box attack tests
struct DummyBlackBoxModel <: AbstractModel end
AdversarialAttacks.predict(::DummyBlackBoxModel, x) = x

@testset "BasicRandomSearch Struct" begin

    # Test default constructor
    attack = BasicRandomSearch()
    @test attack isa BasicRandomSearch
    @test attack.parameters == Dict{String,Any}()

    # Test constructor with parameters
    params = Dict{String,Any}("epsilon" => 0.25)
    attack_with_params = BasicRandomSearch(params)
    @test attack_with_params isa BasicRandomSearch
    @test attack_with_params.parameters == params

    # Test type hierarchy
    @test BasicRandomSearch <: BlackBoxAttack
    @test BasicRandomSearch <: AbstractAttack

    sample = [1.0, 2.0, 3.0]
    model = DummyBlackBoxModel()

    result = craft(sample, model, attack_with_params)
    @test result == sample
    @test size(result) == size(sample)
    @test eltype(result) == eltype(sample)

end

@testset "SquareAttack Struct" begin

    # Test default constructor
    attack = SquareAttack()
    @test attack isa SquareAttack
    @test attack.parameters == Dict{String,Any}()

    # Test constructor with parameters
    params = Dict{String,Any}("epsilon" => 0.25)
    attack_with_params = SquareAttack(params)
    @test attack_with_params isa SquareAttack
    @test attack_with_params.parameters == params

    # Test type hierarchy
    @test SquareAttack <: BlackBoxAttack
    @test SquareAttack <: AbstractAttack

    sample = [1.0, 2.0, 3.0]
    model = DummyBlackBoxModel()

    result = craft(sample, model, attack_with_params)
    @test result == sample
    @test size(result) == size(sample)
    @test eltype(result) == eltype(sample)

end
