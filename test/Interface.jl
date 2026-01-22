module InterfaceTests

using Test
using AdversarialAttacks
using Flux
using DecisionTree

@testset "Interface module" begin

    struct DummyAttack <: AbstractAttack end
    # dummy craft dispatch
    AdversarialAttacks.craft(sample, model, ::DummyAttack; kwargs...) = sample .+ 1.0

    @testset "Invalid inputs" begin
        struct NotAnAttack end
        struct NotAModel end

        @test_throws MethodError attack(NotAnAttack(), Chain(Dense(1, 1)), [1, 2, 3])
        @test_throws MethodError attack(DummyAttack(), Chain(Dense(1, 1)), "asdf")
    end

    @testset "Sample tests" begin
        sample = [1.0, 2.0]
        @test attack(DummyAttack(), Chain(Dense(2, 2)), sample) == [2.0, 3.0]

        matrix = ones(2, 2)
        @test attack(DummyAttack(), Chain(Dense(4, 4)), matrix) == [2.0 2.0; 2.0 2.0]

        tensor = ones(2, 2, 3)
        result = attack(DummyAttack(), Chain(x -> x), tensor)
        @test result == fill(2.0, 2, 2, 3)
    end

    @testset "WhiteBox / BlackBox dispatch" begin
        struct DummyWB <: WhiteBoxAttack end
        struct DummyBB <: BlackBoxAttack end

        struct DummyModel end
        (m::DummyModel)(x) = x

        # minimal dummy craft methods to test Interface.attack dispatch behavior
        AdversarialAttacks.craft(sample, ::DummyModel, ::DummyWB; kwargs...) = sample .* 2
        AdversarialAttacks.craft(sample, ::DummyModel, ::DummyBB; kwargs...) = sample .* 3

        @test attack(DummyWB(), DummyModel(), [1.0, 2.0]) == [2.0, 4.0]
        @test attack(DummyBB(), DummyModel(), [1.0, 2.0]) == [3.0, 6.0]
    end

    @testset "WhiteBox / BlackBox dispatch with NamedTuple" begin
        struct MockWB <: WhiteBoxAttack end
        struct MockBB <: BlackBoxAttack end

        model = Chain(Dense(2 => 2), softmax)

        # WhiteBox + Flux.Chain + NamedTuple
        AdversarialAttacks.craft(sample::NamedTuple, m::Flux.Chain, ::MockWB) = sample.data .* 2.0

        sample_wb = (data=[1.0, 2.0], label=1)
        @test attack(MockWB(), model, sample_wb) == [2.0, 4.0]

        # BlackBox + Flux.Chain + NamedTuple
        AdversarialAttacks.craft(sample::NamedTuple, m::Flux.Chain, ::MockBB) = sample.data .* 1.5

        sample_bb = (data=[1.0, 2.0], label=1)
        @test attack(MockBB(), model, sample_bb) == [1.5, 3.0]
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
        struct DummyBB2 <: BlackBoxAttack end

        X = [0.0 1.0; 1.0 0.0]
        y = [1, 2]
        tree = DecisionTreeClassifier(; max_depth=2)
        fit!(tree, X, y)

        AdversarialAttacks.craft(sample::AbstractArray, ::DecisionTreeClassifier, ::DummyBB2) = sample .* 1.1
        AdversarialAttacks.craft(sample::NamedTuple, ::DecisionTreeClassifier, ::DummyBB2) = sample.data .* 1.1

        sample_vec = [1.0, 2.0]

        # Raw vector
        @test attack(DummyBB2(), tree, sample_vec) ≈ [1.1, 2.2] atol = 1e-6

        # NamedTuple (Int label)
        sample_nt = (data=[1.0, 2.0], label=1)
        @test attack(DummyBB2(), tree, sample_nt) ≈ [1.1, 2.2] atol = 1e-6

        # NamedTuple (Vector{Int} label)
        sample_vec_label = (data=[3.0, 4.0], label=[2])
        @test attack(DummyBB2(), tree, sample_vec_label) ≈ [3.3, 4.4] atol = 1e-6
    end


    @testset "Benchmark" begin
        dataset = [([1.0], 0), ([2.0], 1)]
        function metric(model, adv_samples, labels)
            @test length(adv_samples) == length(labels)
            return length(adv_samples)
        end
        result = benchmark(DummyAttack(), Chain(x -> x), dataset, metric)
        @test result == 2
    end
end
end # module
