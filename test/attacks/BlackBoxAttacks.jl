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
    @test attack.epsilon == 0.1
    @test attack.bounds === nothing

    # Test constructor with parameters
    attack_with_params = BasicRandomSearch(epsilon=0.25)
    @test attack_with_params isa BasicRandomSearch
    @test attack_with_params.epsilon == 0.25
    @test attack_with_params.bounds === nothing

    # Test constructor with bounds
    bounds = [(0.0, 1.0), (0.0, 2.0)]
    attack_with_bounds = BasicRandomSearch(epsilon=0.1, bounds=bounds)
    @test attack_with_bounds.epsilon == 0.1
    @test attack_with_bounds.bounds == bounds

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
    @test result != sample.data  # epsilon > 0, so perturbation expected
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
    atk = BasicRandomSearch(epsilon=ε)

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

@testset "Custom Bounds Support" begin
    Random.seed!(1234)

    # Iris-like bounds: per-feature [lb, ub]
    iris_bounds = [(4.3f0, 7.9f0), (2.0f0, 4.4f0), (1.0f0, 6.9f0), (0.1f0, 2.5f0)]

    # Test 1: Bounds stored correctly in struct
    attack_bounds = BasicRandomSearch(epsilon=0.1f0, bounds=iris_bounds)
    @test attack_bounds.bounds == iris_bounds
    @test attack_bounds.epsilon == 0.1f0

    # Test 2: Default bounds is nothing
    attack_default = BasicRandomSearch(epsilon=0.1f0)
    @test attack_default.bounds === nothing

    # Test 3: SumModel with custom bounds - left direction clamped correctly
    struct BoundedSumModel <: AbstractModel end
    AdversarialAttacks.predict(::BoundedSumModel, x) = x
    function Base.getproperty(::BoundedSumModel, s::Symbol)
        s === :model && return x -> Float32[sum(x), 0.0f0]  # minimize sum(x)
        return getfield(BoundedSumModel, s)
    end

    # Sample near lower bound, expect clamping
    sample_near_lb = (data=Float32[4.4, 2.1, 1.1, 0.2], label=1)  # just above iris_bounds lb
    attack_near_lb = BasicRandomSearch(epsilon=0.2f0, bounds=iris_bounds)
    adv_near_lb = craft(sample_near_lb, BoundedSumModel(), attack_near_lb)

    @test all(adv_near_lb .>= [4.3, 2.0, 1.0, 0.1])     # >= lb
    @test all(adv_near_lb .<= [7.9, 4.4, 6.9, 2.5])     # <= ub
    @test any(adv_near_lb .< sample_near_lb.data)       # decreased (success)

    # Test 4: No bounds → [0,1] default (image compatibility)
    attack_no_bounds = BasicRandomSearch(epsilon=0.1f0)
    sample_image = (data=Float32[0.2, 0.8, 0.3, 0.9], label=1)
    adv_image = craft(sample_image, BoundedSumModel(), attack_no_bounds)
    @test all(0 .<= adv_image .<= 1)                    # [0,1] respected
    @test any(adv_image .< sample_image.data)           # perturbation applied

    # Test 5: Bounds length validation (should error on mismatch)
    invalid_bounds = [(0.0f0, 1.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0)]  # 3 bounds for 4 features
    @test_throws DimensionMismatch craft(
        sample_near_lb,
        BoundedSumModel(),
        BasicRandomSearch(epsilon=0.1f0, bounds=invalid_bounds),
    )
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
