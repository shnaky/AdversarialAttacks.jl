using Test
using AdversarialAttacks
using Random
using Flux

# Shared dummy model for black-box attack tests
struct DummyBlackBoxModel <: AbstractModel end
AdversarialAttacks.predict(::DummyBlackBoxModel, x) = x

@testset "BasicRandomSearch Struct" begin
    Random.seed!(1234)  # Make tests deterministic

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

    # Test for FluxModel
    sample = (data=Float32[1.0, 2.0, 3.0, 7.0], label=Flux.onehot(1, 1:2))
    Random.seed!(1234)
    model = FluxModel(Chain(Dense(4 => 2)))
    x_copy = copy(sample.data)

    result = craft(sample, model, attack_with_params)
    @test result isa Vector
    @test size(result) == size(sample.data)
    @test eltype(result) == eltype(sample.data)
    @test x_copy == sample.data
    @test result != sample.data #as we know epsilon is larger 0 here
    # TODO: Add tests for different models


end

@testset "BasicRandomSearch SimBA core behavior" begin
    Random.seed!(1234)

    # Dummy model 1: model.model(x) = [sum(x), 0.0]
    # For label = 1, decreasing sum(x) decreases true-class "probability"
    struct SumModel <: AbstractModel end
    AdversarialAttacks.predict(::SumModel, x) = x
    function Base.getproperty(::SumModel, s::Symbol)
        s === :model && return x -> Float32[sum(x), 0.0f0]
        return getfield(SumModel, s)
    end

    ε = 0.1f0
    atk = BasicRandomSearch(Dict("epsilon" => ε))

    # Case 1: left move should be chosen at least once
    sample_left = (data=Float32[0.5, 0.5, 0.5, 0.5], label=1)
    model_left = SumModel()

    adv_left = craft(sample_left, model_left, atk)
    @test size(adv_left) == size(sample_left.data)
    @test all(0 .<= adv_left .<= 1)
    @test any(adv_left .< sample_left.data)   # some coordinate decreased
    @test any(adv_left .!= sample_left.data)  # at least one coordinate changed

    # Dummy model 2: model.model(x) = [-sum(x), 0.0]
    # For label = 1, increasing sum(x) decreases true-class "probability"
    struct NegSumModel <: AbstractModel end
    AdversarialAttacks.predict(::NegSumModel, x) = x
    function Base.getproperty(::NegSumModel, s::Symbol)
        s === :model && return x -> Float32[-sum(x), 0.0f0]
        return getfield(NegSumModel, s)
    end

    sample_right = (data=Float32[0.5, 0.5, 0.5, 0.5], label=1)
    model_right = NegSumModel()

    adv_right = craft(sample_right, model_right, atk)
    @test size(adv_right) == size(sample_right.data)
    @test all(0 .<= adv_right .<= 1)
    @test any(adv_right .> sample_right.data) # some coordinate increased
    @test any(adv_right .!= sample_right.data)

    # Dummy model 3: model.model(x) = [0.0, 0.0]
    # No perturbation improves the probability, so x should stay unchanged
    struct ConstModel <: AbstractModel end
    AdversarialAttacks.predict(::ConstModel, x) = x
    function Base.getproperty(::ConstModel, s::Symbol)
        s === :model && return x -> Float32[0.0f0, 0.0f0]
        return getfield(ConstModel, s)
    end

    sample_const = (data=Float32[0.3, 0.7, 0.2, 0.9], label=1)
    model_const = ConstModel()

    adv_const = craft(sample_const, model_const, atk)
    @test adv_const == sample_const.data      # no change if prob is constant
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
