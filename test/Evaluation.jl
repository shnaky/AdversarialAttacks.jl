using Test
using AdversarialAttacks
using Flux
using MLJ
using MLJ: fit!
using CategoricalArrays: levels

@testset "Evaluation Suite" begin
    model = Chain(Dense(4, 3), softmax)

    test_data = [
        (data = randn(Float32, 4), label = Flux.onehot(rand(1:3), 1:3))
            for _ in 1:10
    ]
    attack = FGSM(epsilon = 0.1)

    @testset "evaluate_robustness - basic functionality" begin
        result = evaluate_robustness(model, attack, test_data; num_samples = 5)

        @test hasfield(RobustnessReport, :num_samples)
        @test hasfield(RobustnessReport, :clean_accuracy)
        @test hasfield(RobustnessReport, :adv_accuracy)
        @test hasfield(RobustnessReport, :attack_success_rate)
        @test hasfield(RobustnessReport, :robustness_score)
        @test hasfield(RobustnessReport, :num_successful_attacks)
        @test hasfield(RobustnessReport, :num_clean_correct)

        # L_inf norm fields
        @test hasfield(RobustnessReport, :linf_norm_max)
        @test hasfield(RobustnessReport, :linf_norm_mean)
        @test hasfield(RobustnessReport, :l2_norm_max)
        @test hasfield(RobustnessReport, :l2_norm_mean)
        @test hasfield(RobustnessReport, :l1_norm_max)
        @test hasfield(RobustnessReport, :l1_norm_mean)

        @test result.linf_norm_max isa Float64
        @test result.linf_norm_mean isa Float64
        @test result.linf_norm_max >= 0.0
        @test result.linf_norm_mean >= 0.0
        @test result.linf_norm_max >= result.linf_norm_mean

        @test result.l2_norm_max isa Float64
        @test result.l2_norm_mean isa Float64
        @test result.l2_norm_max >= 0.0
        @test result.l2_norm_mean >= 0.0
        @test result.l2_norm_max >= result.l2_norm_mean

        @test result.l1_norm_max isa Float64
        @test result.l1_norm_mean isa Float64
        @test result.l1_norm_max >= 0.0
        @test result.l1_norm_mean >= 0.0
        @test result.l1_norm_max >= result.l1_norm_mean

        @test result.num_samples isa Int
        @test result.num_clean_correct isa Int
        @test result.clean_accuracy isa Float64
        @test result.adv_accuracy isa Float64
        @test result.attack_success_rate isa Float64
        @test result.robustness_score isa Float64
        @test result.num_successful_attacks isa Int

        @test 0.0 <= result.clean_accuracy <= 1.0
        @test 0.0 <= result.adv_accuracy <= 1.0
        @test 0.0 <= result.attack_success_rate <= 1.0
        @test 0.0 <= result.robustness_score <= 1.0
        @test result.num_samples == 5

        # Test that success_rate and robustness_score sum to 1.0
        @test result.attack_success_rate + result.robustness_score ≈ 1.0
    end

    @testset "evaluate_robustness - calculate_metrics" begin
        n_test = 10
        num_clean_correct = 0
        num_adv_correct = 5
        num_successful_attacks = 5
        l_norms = Dict(
            :linf => Float64[],
            :l2 => Float64[],
            :l1 => Float64[]
        )
        metrics = AdversarialAttacks.calculate_metrics(
            n_test,
            num_clean_correct,
            num_adv_correct,
            num_successful_attacks,
            l_norms
        )
        @test metrics isa RobustnessReport

        # else cases
        @test metrics.attack_success_rate == 0.0
        @test metrics.linf_norm_max == metrics.linf_norm_mean == 0.0
        @test metrics.l2_norm_max == metrics.l2_norm_mean == 0.0
        @test metrics.l1_norm_max == metrics.l1_norm_mean == 0.0

        # if cases
        num_clean_correct = 5
        l_norms[:linf] = [0.5]
        l_norms[:l2] = [0.5]
        l_norms[:l1] = [0.5]

        metrics = AdversarialAttacks.calculate_metrics(
            n_test,
            num_clean_correct,
            num_adv_correct,
            num_successful_attacks,
            l_norms
        )
        @test metrics.clean_accuracy == num_clean_correct / n_test
        @test metrics.adv_accuracy == num_adv_correct / num_clean_correct
        @test metrics.attack_success_rate == num_successful_attacks / num_clean_correct
        @test metrics.robustness_score == 1.0 - metrics.attack_success_rate
        @test metrics.linf_norm_max > 0.0 && metrics.linf_norm_mean > 0.0
        @test metrics.l2_norm_max > 0.0 && metrics.l2_norm_mean > 0.0
        @test metrics.l1_norm_max > 0.0 && metrics.l1_norm_mean > 0.0
    end

    @testset "evaluate_robustness - compute_norm" begin
        sample_data = [1.0, 2.0, 3.0]
        adv_data = sample_data .* 2.0
        linf = AdversarialAttacks.compute_norm(sample_data, adv_data, Inf)
        l2 = AdversarialAttacks.compute_norm(sample_data, adv_data, 2)
        l1 = AdversarialAttacks.compute_norm(sample_data, adv_data, 1)
        @test linf == 3.0
        @test isapprox(l2, sqrt(14); rtol = 1.0e-6)
        @test l1 == 6.0
    end

    @testset "evaluate_robustness - num_samples handling" begin
        # Should use all available samples (10) instead of requested (20)
        result = evaluate_robustness(model, attack, test_data; num_samples = 20)
        @test result.num_samples == 10

        # Should only process the requested number of samples
        result = evaluate_robustness(model, attack, test_data; num_samples = 3)
        @test result.num_samples == 3
    end

    @testset "evaluate_robustness - edge cases" begin
        @test_throws ArgumentError evaluate_robustness(
            nothing, nothing, test_data; num_samples = 0
        )

        @test_throws ArgumentError evaluate_robustness(
            nothing, nothing, test_data; num_samples = -1
        )

        @test_warn "Failed to evaluate sample 1" evaluate_robustness(
            nothing, nothing, test_data; num_samples = 1
        )
    end

    @testset "evaluate_robustness - RobustnessReport show" begin
        # dummy report
        report = RobustnessReport(
            1,      # n_test
            0,      # num_clean_correct
            0.0,    # clean_accuracy
            0.0,    # adv_accuracy
            0.0,    # attack_success_rate
            1.0,    # robustness_score
            0,      # num_successful_attacks
            0.0,    # linf_norm_max
            0.0,    # linf_norm_mean
            0.0,    # l2_norm_max
            0.0,    # l2_norm_mean
            0.0,    # l1_norm_max
            0.0     # l1_norm_mean
        )

        # Capture output from Base.show
        io = IOBuffer()
        show(io, report)
        output = String(take!(io))

        # Verify key sections appear in output
        @test occursin("Robustness Evaluation Report", output)
        @test occursin("Total samples evaluated", output)
        @test occursin("Perturbation Analysis (Norms)", output)

        # Verify all norm types are displayed
        @test occursin("L_inf Maximum perturbation", output)
        @test occursin("L_inf Mean perturbation", output)
        @test occursin("L_2 Maximum perturbation", output)
        @test occursin("L_2 Mean perturbation", output)
        @test occursin("L_1 Maximum perturbation", output)
        @test occursin("L_1 Mean perturbation", output)
    end

    @testset "evaluate_robustness - L_inf norm correctness" begin
        # simple attack that adds a fixed perturbation
        struct TestLinfAttack end
        AdversarialAttacks.attack(::TestLinfAttack, model, sample) = sample.data .+ Float32[0.1, 0.5, 0.2, 0.3]

        # single test sample with an anonymous dummy function as model
        test_data = [(data = Float32[1, 2, 3, 4], label = Flux.onehot(1, 1:3))]
        result = evaluate_robustness(x -> Flux.onehot(1, 1:3), TestLinfAttack(), test_data; num_samples = 1)

        # check L_inf norms
        @test result.linf_norm_max == 0.5
        @test result.linf_norm_mean == 0.5
    end

    @testset "evaluate_robustness - evaluation_curve" begin
        attack_type = FGSM

        epsilons = [0.05, 0.1]

        results = evaluation_curve(model, attack_type, epsilons, test_data; num_samples = 5)

        @test results isa Dict
        @test results[:epsilons] == epsilons
        @test length(results[:clean_accuracy]) == 2
        @test length(results[:linf_norm_mean]) == 2
    end

    @testset "make_prediction_function" begin
        @testset "Flux-like model branch" begin
            # Test Flux-style neural network prediction wrapper
            model = Chain(Dense(4, 3), softmax)
            f = AdversarialAttacks.make_prediction_function(model)

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
            mach = machine(model, X, y) |> fit!

            # Test MLJ prediction wrapper with vector input
            f = AdversarialAttacks.make_prediction_function(mach)
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
            mach = machine(model, X, y) |> fit!

            f = AdversarialAttacks.make_prediction_function(mach)

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
end
