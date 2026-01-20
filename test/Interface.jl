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

    @testset "WhiteBox / BlackBox dispatch with NamedTuple" begin
        struct MockDiffModel <: DifferentiableModel end
        struct MockWB <: WhiteBoxAttack end
        struct MockBB <: BlackBoxAttack end

        # WhiteBox + DifferentiableModel + NamedTuple
        AdversarialAttacks.craft(sample::NamedTuple, ::MockDiffModel, ::MockWB) = sample.data .* 2.0
        sample_wb = (data=[1.0, 2.0], label=1)
        @test attack(MockWB(), MockDiffModel(), sample_wb) == [2.0, 4.0]

        # BlackBox + DifferentiableModel + NamedTuple
        AdversarialAttacks.craft(sample::NamedTuple, ::MockDiffModel, ::MockBB) = sample.data .* 1.5
        sample_bb = (data=[1.0, 2.0], label=1)
        @test attack(MockBB(), MockDiffModel(), sample_bb) == [1.5, 3.0]
    end

    @testset "DecisionTree dispatch" begin
        struct DummyBBTree <: BlackBoxAttack end

        # Define craft for DecisionTreeClassifier with the dummy attack
        AdversarialAttacks.craft(sample, ::DecisionTreeClassifier, ::DummyBBTree) = sample .+ 5.0
        AdversarialAttacks.craft(sample::NamedTuple, ::DecisionTreeClassifier, ::DummyBBTree) = sample.data .+ 5.0

        # Create a minimal DecisionTreeClassifier
        X = [0.0 1.0; 1.0 0.0]  # 2 samples, 2 features
        y = [1, 2]              # 1-based labels

        tree = DecisionTreeClassifier(; max_depth=2)
        fit!(tree, X, y)

        # Test AbstractArray sample
        raw_sample = [1.0, 2.0]
        adv_raw = attack(DummyBBTree(), tree, raw_sample)
        @test adv_raw == raw_sample .+ 5.0

        # Test NamedTuple sample
        nt_sample = (data=[1.0, 2.0], label=1)
        adv_nt = attack(DummyBBTree(), tree, nt_sample)
        @test adv_nt == nt_sample.data .+ 5.0
    end

    @testset "TreeModel dispatch with NamedTuple" begin
        struct MockTreeModel <: AbstractModel end
        struct DummyBB <: BlackBoxAttack end

        AdversarialAttacks.craft(sample::AbstractArray, ::MockTreeModel, ::DummyBB) = sample .* 1.1
        AdversarialAttacks.craft(sample::NamedTuple, ::MockTreeModel, ::DummyBB) = sample.data .* 1.1

        sample_vec = [1.0, 2.0]

        # Raw vector
        @test attack(DummyBB(), MockTreeModel(), sample_vec) ≈ [1.1, 2.2] atol = 1e-6

        # NamedTuple (Int label)
        sample_nt = (data=[1.0, 2.0], label=1)
        @test attack(DummyBB(), MockTreeModel(), sample_nt) ≈ [1.1, 2.2] atol = 1e-6

        # NamedTuple (Vector{Int} label)
        sample_vec_label = (data=[3.0, 4.0], label=[2])
        @test attack(DummyBB(), MockTreeModel(), sample_vec_label) ≈ [3.3, 4.4] atol = 1e-6
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
end
end # module
