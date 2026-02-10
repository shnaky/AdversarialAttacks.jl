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

    @testset "DecisionTreeClassifier branch" begin
        # Binary classification training data
        X_train = randn(50, 4)
        y_train = categorical(rand([0, 1], 50))

        tree_model = DecisionTreeClassifier()
        dt_fit!(tree_model, X_train, y_train)

        f = make_prediction_function(tree_model)

        # Vector input → 1×4 reshape → 1×2 Matrix output
        x_vec = randn(4)
        probs_vec = f(x_vec)
        @test probs_vec isa AbstractMatrix{Float64}
        @test size(probs_vec) == (1, 2)
        @test all(0 .≤ probs_vec .≤ 1)
        @test isapprox(sum(probs_vec), 1; atol = 1.0e-6)

        # Matrix input → vec(x_flat) → Float64.(...) → 1×4 → 1×2 Matrix
        x_mat = randn(Float32, 2, 2)
        probs_mat = f(x_mat)
        @test probs_mat isa AbstractMatrix{Float64}
        @test size(probs_mat) == (1, 2)

        # To get vec for unified interface (like MLJ/Flux)
        probs_vec_from_mat = vec(probs_mat)
        @test probs_vec_from_mat isa AbstractVector{Float64}
        @test length(probs_vec_from_mat) == 2
    end
end
