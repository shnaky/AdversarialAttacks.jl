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
    atk = BasicRandomSearch()
    @test atk isa BasicRandomSearch
    @test atk.epsilon == 0.1
    @test atk.bounds === nothing

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

    # Test constructor with epsilon, bounds, and max_iter
    attack_full = BasicRandomSearch(epsilon = 0.2, bounds = bounds, max_iter = 50)
    @test attack_full.max_iter == 50

    # Test type hierarchy
    @test BasicRandomSearch <: BlackBoxAttack
    @test BasicRandomSearch <: AbstractAttack

    # Test for Flux model
    sample = (data = Float32[1.0, 2.0, 3.0, 7.0], label = Flux.onehot(1, 1:2))
    Random.seed!(1234)
    model = Chain(Dense(4 => 2), softmax)
    x_copy = copy(sample.data)

    result = attack(attack_with_params, model, sample)
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

    # Test attack with typed API
    sample = (data = Float32[0.5, 0.8, 1.2, 1.0], label = Flux.onehot(1, 1:3))
    atk = BasicRandomSearch(epsilon = 0.1f0)
    x_copy = copy(sample.data)

    result = attack(atk, dt_model, sample)
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

    adv_left = attack(atk, model_left, sample_left)
    @test size(adv_left) == size(sample_left.data)
    @test all(0 .<= adv_left .<= 1)
    @test any(adv_left .< sample_left.data)   # some coordinate decreased
    @test any(adv_left .!= sample_left.data)  # at least one coordinate changed

    # Case 2: right move should be chosen at least once
    # Dummy Flux model 2: output = [-sum(x), 0.0]
    # For label = 1, increasing sum(x) decreases true-class "probability".
    model_right = Chain(x -> Float32[-sum(x), 0.0f0])

    sample_right = (data = Float32[0.5, 0.5, 0.5, 0.5], label = 1)

    adv_right = attack(atk, model_right, sample_right)
    @test size(adv_right) == size(sample_right.data)
    @test all(0 .<= adv_right .<= 1)
    @test any(adv_right .> sample_right.data) # some coordinate increased
    @test any(adv_right .!= sample_right.data)

    # Case 3: no move should be taken if probabilities are constant
    # Dummy Flux model 3: output = [0.0, 0.0]
    model_const = Chain(x -> Float32[0.0f0, 0.0f0])

    sample_const = (data = Float32[0.3, 0.7, 0.2, 0.9], label = 1)

    adv_const = attack(atk, model_const, sample_const)
    @test adv_const == sample_const.data      # no change if prob is constant

    # Case 4: verify early stopping works with a simple model
    # Model where class 1 prob decreases with each perturbation
    model_early_stop = Chain(x -> Float32[sum(x), 10.0f0 - sum(x)])

    sample_early = (data = Float32[2.0, 2.0, 1.0, 1.0], label = 1)  # sum = 6.0, class 1 initially predicted

    # Run with sufficient iterations to guarantee misclassification
    atk_early = BasicRandomSearch(epsilon = 0.3f0, max_iter = 100)
    adv_early = attack(atk_early, model_early_stop, sample_early)

    # Verify that misclassification was achieved
    # If early stopping works correctly, it stops at the first misclassification, should happen before 100 iterations
    @test argmax(model_early_stop(adv_early)) != 1
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
    adv_near_lb = attack(attack_near_lb, bounded_model, sample_near_lb)

    @test all(adv_near_lb .>= [4.3, 2.0, 1.0, 0.1])     # >= lb
    @test all(adv_near_lb .<= [7.9, 4.4, 6.9, 2.5])     # <= ub
    @test any(adv_near_lb .< sample_near_lb.data)       # decreased (success)

    # Test 4: No bounds → [0,1] default (image compatibility)
    attack_no_bounds = BasicRandomSearch(epsilon = 0.1f0)
    sample_image = (data = Float32[0.2, 0.8, 0.3, 0.9], label = 1)
    adv_image = attack(attack_no_bounds, bounded_model, sample_image)
    @test all(0 .<= adv_image .<= 1)                    # [0,1] respected
    @test any(adv_image .< sample_image.data)           # perturbation applied

    # Test 5: Bounds length validation (should error on mismatch)
    invalid_bounds = [(0.0f0, 1.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0)]  # 3 bounds for 4 features
    @test_throws DimensionMismatch attack(
        BasicRandomSearch(epsilon = 0.1f0, bounds = invalid_bounds),
        bounded_model,
        sample_near_lb,
    )
end

@testset "Custom Max Iterations" begin
    Random.seed!(1234)

    # Dummy Flux model: output = [sum(x), 0.0]
    model = Chain(x -> Float32[sum(x), 0.0f0])

    sample = (data = Float32[0.5, 0.5, 0.5, 0.5], label = 1)

    # Test with max_iter = 5
    atk_low_iter = BasicRandomSearch(epsilon = 0.1f0, max_iter = 5)
    adv_low_iter = attack(atk_low_iter, model, sample)
    @test size(adv_low_iter) == size(sample.data)

    # Test with max_iter = 50
    atk_high_iter = BasicRandomSearch(epsilon = 0.1f0, max_iter = 50)
    adv_high_iter = attack(atk_high_iter, model, sample)
    @test size(adv_high_iter) == size(sample.data)

    # Check that higher max_iter can lead to larger perturbations
    perturb_low = sum(abs.(adv_low_iter .- sample.data))
    perturb_high = sum(abs.(adv_high_iter .- sample.data))
    @test perturb_high >= perturb_low
end
