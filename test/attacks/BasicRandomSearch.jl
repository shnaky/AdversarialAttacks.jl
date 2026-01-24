using Test
using AdversarialAttacks
using Random
using Flux
using DecisionTree

# Shared dummy model for black-box attack tests
struct DummyBlackBoxModel end

@testset "BasicRandomSearch Struct" begin
    Random.seed!(1234)  # Make tests deterministic

    # Test default constructor
    attack = BasicRandomSearch()
    @test attack isa BasicRandomSearch
    @test attack.epsilon == 0.1
    @test attack.bounds === nothing

    # Test constructor with parameters
    attack_with_params = BasicRandomSearch(epsilon = 0.25)
    @test attack_with_params isa BasicRandomSearch
    @test attack_with_params.epsilon == 0.25
    @test attack_with_params.bounds === nothing

    # Test constructor with bounds
    bounds = [(0.0, 1.0), (0.0, 2.0)]
    attack_with_bounds = BasicRandomSearch(epsilon = 0.1, bounds = bounds)
    @test attack_with_bounds.epsilon == 0.1
    @test attack_with_bounds.bounds == bounds

    # Test type hierarchy
    @test BasicRandomSearch <: BlackBoxAttack
    @test BasicRandomSearch <: AbstractAttack

    # Test for Flux model
    sample = (data = Float32[1.0, 2.0, 3.0, 7.0], label = Flux.onehot(1, 1:2))
    Random.seed!(1234)
    model = Chain(Dense(4 => 2), softmax)
    x_copy = copy(sample.data)

    result = craft(sample, model, attack_with_params)
    @test result isa Vector
    @test size(result) == size(sample.data)
    @test eltype(result) == eltype(sample.data)
    @test x_copy == sample.data
end

@testset "BasicRandomSearch with DecisionTreeClassifier" begin
    Random.seed!(1234)

    classes = ["A", "B", "C"]
    labels = vcat(fill(classes[1], 8), fill(classes[2], 8), fill(classes[3], 8))
    features = rand(24, 4) .* 4

    dt_model = DecisionTreeClassifier(; classes = classes)
    fit!(dt_model, features, labels)

    # Verify predict_proba returns 3 probs
    test_x = reshape(Float64[1.0, 1.5, 2.0, 2.5], 1, 4)
    probs = predict_proba(dt_model, test_x)
    @test length(probs) == 3
    @test all(0 .<= probs .<= 1)

    # Test craft with typed API
    sample = (data = Float32[0.5, 0.8, 1.2, 1.0], label = Flux.onehot(1, 1:3))
    attack = BasicRandomSearch(epsilon = 0.1f0)
    x_copy = copy(sample.data)

    result = craft(sample, dt_model, attack)
    @test result isa Vector{Float32}
    @test size(result) == size(sample.data)
    @test x_copy == sample.data  # Original unchanged
end

@testset "BasicRandomSearch SimBA core behavior" begin
    Random.seed!(1234)

    ε = 0.1f0
    atk = BasicRandomSearch(epsilon = ε)

    # Case 1: left move should be chosen at least once
    # Dummy Flux model 1: output = [sum(x), 0.0]
    # For label = 1, decreasing sum(x) decreases true-class "probability".
    model_left = Chain(x -> Float32[sum(x), 0.0f0])

    sample_left = (data = Float32[0.5, 0.5, 0.5, 0.5], label = 1)

    adv_left = craft(sample_left, model_left, atk)
    @test size(adv_left) == size(sample_left.data)
    @test all(0 .<= adv_left .<= 1)
    @test any(adv_left .< sample_left.data)   # some coordinate decreased
    @test any(adv_left .!= sample_left.data)  # at least one coordinate changed

    # Case 2: right move should be chosen at least once
    # Dummy Flux model 2: output = [-sum(x), 0.0]
    # For label = 1, increasing sum(x) decreases true-class "probability".
    model_right = Chain(x -> Float32[-sum(x), 0.0f0])

    sample_right = (data = Float32[0.5, 0.5, 0.5, 0.5], label = 1)

    adv_right = craft(sample_right, model_right, atk)
    @test size(adv_right) == size(sample_right.data)
    @test all(0 .<= adv_right .<= 1)
    @test any(adv_right .> sample_right.data) # some coordinate increased
    @test any(adv_right .!= sample_right.data)

    # Case 3: no move should be taken if probabilities are constant
    # Dummy Flux model 3: output = [0.0, 0.0]
    model_const = Chain(x -> Float32[0.0f0, 0.0f0])

    sample_const = (data = Float32[0.3, 0.7, 0.2, 0.9], label = 1)

    adv_const = craft(sample_const, model_const, atk)
    @test adv_const == sample_const.data      # no change if prob is constant
end

@testset "Custom Bounds Support" begin
    Random.seed!(1234)

    # Iris-like bounds: per-feature [lb, ub]
    iris_bounds = [(4.3f0, 7.9f0), (2.0f0, 4.4f0), (1.0f0, 6.9f0), (0.1f0, 2.5f0)]

    # Test 1: Bounds stored correctly in struct
    attack_bounds = BasicRandomSearch(epsilon = 0.1f0, bounds = iris_bounds)
    @test attack_bounds.bounds == iris_bounds
    @test attack_bounds.epsilon == 0.1f0

    # Test 2: Default bounds is nothing
    attack_default = BasicRandomSearch(epsilon = 0.1f0)
    @test attack_default.bounds === nothing

    # Dummy Flux model with custom bounds: model(x) = [sum(x), 0.0]
    # For label = 1, decreasing sum(x) decreases the true-class "probability".
    bounded_model = Chain(x -> Float32[sum(x), 0.0f0])

    # Sample near lower bound, expect clamping
    sample_near_lb = (data = Float32[4.4, 2.1, 1.1, 0.2], label = 1)  # just above iris_bounds lb
    attack_near_lb = BasicRandomSearch(epsilon = 0.2f0, bounds = iris_bounds)
    adv_near_lb = craft(sample_near_lb, bounded_model, attack_near_lb)

    @test all(adv_near_lb .>= [4.3, 2.0, 1.0, 0.1])     # >= lb
    @test all(adv_near_lb .<= [7.9, 4.4, 6.9, 2.5])     # <= ub
    @test any(adv_near_lb .< sample_near_lb.data)       # decreased (success)

    # Test 4: No bounds → [0,1] default (image compatibility)
    attack_no_bounds = BasicRandomSearch(epsilon = 0.1f0)
    sample_image = (data = Float32[0.2, 0.8, 0.3, 0.9], label = 1)
    adv_image = craft(sample_image, bounded_model, attack_no_bounds)
    @test all(0 .<= adv_image .<= 1)                    # [0,1] respected
    @test any(adv_image .< sample_image.data)           # perturbation applied

    # Test 5: Bounds length validation (should error on mismatch)
    invalid_bounds = [(0.0f0, 1.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0)]  # 3 bounds for 4 features
    @test_throws DimensionMismatch craft(
        sample_near_lb,
        bounded_model,
        BasicRandomSearch(epsilon = 0.1f0, bounds = invalid_bounds),
    )
end
