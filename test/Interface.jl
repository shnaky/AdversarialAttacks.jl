module InterfaceTests

using Test
using AdversarialAttacks

const Interface = AdversarialAttacks.Interface
const Attack = AdversarialAttacks.Attack
const Model = AdversarialAttacks.Model

@testset "Interface module" begin

    struct DummyAttack <: Attack.AbstractAttack end
    struct DummyModel <: Model.AbstractModel end
    # dummy craft dispatch
    Attack.craft(sample, ::Model.AbstractModel, ::Attack.AbstractAttack; kwargs...) = sample .+ 1.0

    @testset "Invalid inputs" begin
        struct NotAnAttack end
        struct NotAModel end

        @test_throws MethodError Interface.attack(NotAnAttack(), DummyModel(), [1, 2, 3])
        @test_throws MethodError Interface.attack(DummyAttack(), NotAModel(), [1, 2, 3])
        @test_throws MethodError Interface.attack(DummyAttack(), DummyModel(), "asdf")
    end

   @testset "Sample tests" begin
        sample = [1.0, 2.0]
        @test Interface.attack(DummyAttack(), DummyModel(), sample) == [2.0, 3.0]

        matrix = ones((2,2))
        @test Interface.attack(DummyAttack(), DummyModel(), matrix) == [2.0 2.0; 2.0 2.0]

        tensor = ones(2, 2, 3)
        result = Interface.attack(DummyAttack(), DummyModel(), tensor)
        @test result == fill(2.0, 2, 2, 3)
    end

    @testset "WhiteBox / BlackBox dispatch" begin
        WB = Attack.WhiteBoxAttack
        BB = Attack.BlackBoxAttack
        Diff = Model.DifferentiableModel
        NonDiff = Model.NonDifferentiableModel

        struct DummyWB <: WB end
        struct DummyBB <: BB end
        struct DummyDiffModel <: Diff end
        struct DummyNonDiffModel <: NonDiff end

        # minimal dummy craft methods to test Interface.attack dispatch behavior
        Attack.craft(sample, ::Diff, ::WB; kwargs...) = sample .* 2
        Attack.craft(sample, ::NonDiff, ::BB; kwargs...) = sample .* 3

        @test Interface.attack(DummyWB(), DummyDiffModel(), [1.0, 2.0]) == [2.0, 4.0]
        @test Interface.attack(DummyBB(), DummyNonDiffModel(), [1.0, 2.0]) == [3.0, 6.0]
        # whitebox attack with non-differentiable model is not allowed
        @test_throws ErrorException Interface.attack(DummyWB(), DummyNonDiffModel(), [1.0, 2.0])
    end

    @testset "Benchmark" begin
        dataset = [([1.0], 0), ([2.0], 1)]
        function metric(model, adv_samples, labels)
            @test length(adv_samples) == length(labels)
            return length(adv_samples)
        end
        result = Interface.benchmark(DummyAttack(), DummyModel(), dataset, metric)
        @test result == 2
    end
end
end # module
