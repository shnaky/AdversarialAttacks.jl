module InterfaceTests

using Test
using AdversarialAttacks
using DecisionTree

@testset "Interface module" begin

    struct DummyAttack <: AbstractAttack end
    struct DummyModel <: AbstractModel end
    # dummy craft dispatch
    AdversarialAttacks.craft(sample, ::AbstractModel, ::AbstractAttack; kwargs...) = sample .+ 1.0

    @testset "Invalid inputs" begin
        struct NotAnAttack end
        struct NotAModel end

        @test_throws MethodError attack(NotAnAttack(), DummyModel(), [1, 2, 3])
        @test_throws MethodError attack(DummyAttack(), NotAModel(), [1, 2, 3])
        @test_throws MethodError attack(DummyAttack(), DummyModel(), "asdf")
    end

    @testset "Sample tests" begin
        sample = [1.0, 2.0]
        @test attack(DummyAttack(), DummyModel(), sample) == [2.0, 3.0]

        matrix = ones((2, 2))
        @test attack(DummyAttack(), DummyModel(), matrix) == [2.0 2.0; 2.0 2.0]

        tensor = ones(2, 2, 3)
        result = attack(DummyAttack(), DummyModel(), tensor)
        @test result == fill(2.0, 2, 2, 3)
    end

    @testset "WhiteBox / BlackBox dispatch" begin
        struct DummyWB <: WhiteBoxAttack end
        struct DummyBB <: BlackBoxAttack end
        struct DummyDiffModel <: DifferentiableModel end
        struct DummyNonDiffModel <: NonDifferentiableModel end

        # minimal dummy craft methods to test Interface.attack dispatch behavior
        AdversarialAttacks.craft(sample, ::DifferentiableModel, ::WhiteBoxAttack; kwargs...) = sample .* 2
        AdversarialAttacks.craft(sample, ::NonDifferentiableModel, ::BlackBoxAttack; kwargs...) = sample .* 3

        @test attack(DummyWB(), DummyDiffModel(), [1.0, 2.0]) == [2.0, 4.0]
        @test attack(DummyBB(), DummyNonDiffModel(), [1.0, 2.0]) == [3.0, 6.0]
        # whitebox attack with non-differentiable model is not allowed
        @test_throws ErrorException attack(DummyWB(), DummyNonDiffModel(), [1.0, 2.0])
    end

    @testset "Benchmark" begin
        dataset = [([1.0], 0), ([2.0], 1)]
        function metric(model, adv_samples, labels)
            @test length(adv_samples) == length(labels)
            return length(adv_samples)
        end
        result = benchmark(DummyAttack(), DummyModel(), dataset, metric)
        @test result == 2
    end

    @testset "DecisionTree dispatch" begin
        struct DummyBBTree <: BlackBoxAttack end

        AdversarialAttacks.craft(sample, ::DecisionTreeClassifier, ::BlackBoxAttack; kwargs...) = sample .+ 5.0

        # Minimal DecisionTreeClassifier instance
        X = [0.0 1.0;
            1.0 0.0]            # 2 samples, 2 features
        y = [1, 2]               # 1-based labels

        tree = DecisionTreeClassifier(max_depth=2)
        fit!(tree, X, y)

        raw_sample = [1.0, 2.0]
        adv_raw = attack(DummyBBTree(), tree, raw_sample)
        @test adv_raw == raw_sample .+ 5.0
    end
end
end # module
