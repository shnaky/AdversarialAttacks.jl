@testset "make_prediction_function" begin
    @testset "Flux-like model branch" begin
        # Test Flux-style neural network prediction wrapper
        model = Chain(Dense(4, 3), softmax)
        f = make_prediction_function(model)

        x = randn(Float32, 4)
        probs = f(x)

        # Test vector input (common case)
        @test probs isa AbstractVector
        @test length(probs) == 3
        @test eltype(probs) <: Real
        @test all(p -> 0 ≤ p ≤ 1, probs)
        @test isapprox(sum(probs), 1; atol = 1.0e-5)

        # Test matrix input (batched input case)
        x_mat = randn(Float32, 4, 2)
        probs_mat = f(x_mat)
        @test probs_mat isa AbstractVector
        @test length(probs_mat) == 6  # 3 classes × 2 samples
    end

    @testset "MLJ Machine branch - vector input" begin
        # Create MLJ training data and DecisionTree classifier
        X = (x1 = randn(100), x2 = randn(100), x3 = randn(100), x4 = randn(100))
        y = categorical(rand(1:3, 100))

        Tree = @load DecisionTreeClassifier pkg = DecisionTree verbosity = 0
        model = Tree()
        mach = machine(model, X, y) |> mlj_fit!

        # Test MLJ prediction wrapper with vector input
        f = make_prediction_function(mach)
        x = randn(4)
        probs = f(x)

        @test probs isa AbstractVector
        @test length(probs) == 3
        @test eltype(probs) <: Real
        @test all(p -> 0 ≤ p ≤ 1, probs)
        @test isapprox(sum(probs), 1; atol = 1.0e-5)
    end

    @testset "MLJ Machine branch - array reshaping" begin
        # Test input reshaping logic for non-vector inputs
        X = (x1 = randn(50), x2 = randn(50), x3 = randn(50), x4 = randn(50))
        y = categorical(rand(1:3, 50))

        Tree = @load DecisionTreeClassifier pkg = DecisionTree verbosity = 0
        model = Tree()
        mach = machine(model, X, y) |> mlj_fit!

        f = make_prediction_function(mach)

        # Test 2×2 matrix input (vec → 1×4 reshaping path)
        x_mat = randn(2, 2)
        probs = f(x_mat)
        @test probs isa AbstractVector
        @test length(probs) == 3
        @test all(p -> 0 ≤ p ≤ 1, probs)
        @test isapprox(sum(probs), 1; atol = 1.0e-5)

        # Test 1×4 row vector input (direct reshape path)
        x_row = randn(1, 4)
        probs_row = f(x_row)
        @test length(probs_row) == 3
    end
end
