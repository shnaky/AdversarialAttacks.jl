using Test
using AdversarialAttacks
using Flux

@testset "Evaluation Suite" begin
    model_flux = Chain(Dense(4, 3), softmax)
    model = FluxModel(model_flux)

    test_data = [
        (data=randn(Float32, 4), label=Flux.onehot(rand(1:3), 1:3))
        for _ in 1:10
    ]
    attack = FGSM(epsilon=0.1)

    @testset "evaluate_robustness - basic functionality" begin
        result = evaluate_robustness(model, attack, test_data; num_samples=5)

        @test hasfield(RobustnessReport, :num_samples)
        @test hasfield(RobustnessReport, :clean_accuracy)
        @test hasfield(RobustnessReport, :adv_accuracy)
        @test hasfield(RobustnessReport, :attack_success_rate)
        @test hasfield(RobustnessReport, :robustness_score)
        @test hasfield(RobustnessReport, :num_successful_attacks)
        @test hasfield(RobustnessReport, :num_clean_correct)

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


    @testset "evaluate_robustness - num_samples handling" begin
        # Should use all available samples (10) instead of requested (20)
        result = evaluate_robustness(model, attack, test_data; num_samples=20)
        @test result.num_samples == 10

        # Should only process the requested number of samples
        result = evaluate_robustness(model, attack, test_data; num_samples=3)
        @test result.num_samples == 3
    end

    @testset "evaluate_robustness - edge cases" begin
        @test_throws ArgumentError evaluate_robustness(
            nothing, nothing, test_data; num_samples=0
        )

        @test_throws ArgumentError evaluate_robustness(
            nothing, nothing, test_data; num_samples=-1
        )

        @test_warn "Failed to evaluate sample 1" evaluate_robustness(
            nothing, nothing, test_data; num_samples=1
        )

        result = evaluate_robustness(nothing, nothing, test_data; num_samples=1)
        @test result.num_clean_correct == 0
        @test result.attack_success_rate == 0.0
    end

    @testset "evaluate_robustness - RobustnessReport show" begin
        # dummy report
        report = RobustnessReport(1, 0.0, 0.0, 0.0, 1.0, 0, 0)

        # get output
        io = IOBuffer()
        show(io, report)
        output = String(take!(io))

        # check if key phrases appear in the output
        @test occursin("Robustness Evaluation Report", output)
        @test occursin("Total samples evaluated", output)
    end
end