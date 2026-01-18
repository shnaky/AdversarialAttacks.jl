using Test
using AdversarialAttacks

@testset "Evaluation Suite" begin

    @testset "evaluate_robustness - basic functionality" begin

        test_data = [(data=i, label=i) for i in 1:10]

        # Use nil objects since actual implementation is not ready yet
        model = nothing
        attack = nothing

        result = evaluate_robustness(model, attack, test_data; num_samples=5)

        @test haskey(result, "success_rate")
        @test haskey(result, "robustness_score")
        @test haskey(result, "num_samples")
        @test haskey(result, "num_successful_attacks")

        @test result["success_rate"] isa Float64
        @test result["robustness_score"] isa Float64
        @test result["num_samples"] isa Int
        @test result["num_successful_attacks"] isa Int

        @test 0.0 <= result["success_rate"] <= 1.0
        @test 0.0 <= result["robustness_score"] <= 1.0
        @test result["num_samples"] == 5

        # Test that success_rate and robustness_score sum to 1.0
        @test result["success_rate"] + result["robustness_score"] ≈ 1.0
    end


    @testset "evaluate_robustness - num_samples handling" begin
        test_data = [(data=i, label=i) for i in 1:10]

        # Should use all available samples (10) instead of requested (20)
        result = evaluate_robustness(nothing, nothing, test_data; num_samples=20)
        @test result["num_samples"] == 10

        # Should only process the requested number of samples
        result = evaluate_robustness(nothing, nothing, test_data; num_samples=3)
        @test result["num_samples"] == 3
    end

    @testset "evaluate_robustness - edge cases" begin
        test_data = [(data=i, label=i) for i in 1:10]

        @test_throws ArgumentError evaluate_robustness(
            nothing, nothing, test_data; num_samples=0
        )

        @test_throws ArgumentError evaluate_robustness(
            nothing, nothing, test_data; num_samples=-1
        )
    end

end