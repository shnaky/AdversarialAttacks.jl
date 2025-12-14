module InterfaceTests

using Test
using AdversarialAttacks

const Interface = AdversarialAttacks.Interface
const Attack = AdversarialAttacks.Attack
const Model = AdversarialAttacks.Model

@testset "Interface.jl" begin
    struct DummyAttack <: Attack.AbstractAttack end
    struct DummyModel <: Model.AbstractModel end
    Attack.craft(sample, ::DummyModel, ::DummyAttack; kwargs...) = sample .+ 1.0

    @testset "Fallback run" begin
        struct NotAnAttack end
        struct NotAModel end
        @test_throws MethodError Interface.run(NotAnAttack(), DummyModel(), [1, 2, 3])
        @test_throws MethodError Interface.run(DummyAttack(), NotAModel(), [1, 2, 3])
        @test_throws MethodError Interface.run(DummyAttack(), DummyModel(), "asdf")
    end

    @testset "Single run" begin
        result = Interface.run(DummyAttack(), DummyModel(), [1.0, 2.0])
        @test result == [2.0, 3.0]
    end

    @testset "Batch run" begin
        samples = [[1.0, 2.0], [3.0, 4.0]]
        result = Interface.run(DummyAttack(), DummyModel(), samples)
        @test result == [[2.0, 3.0], [4.0, 5.0]]
    end

    @testset "Benchmark" begin
        dataset = [([1.0], 0), ([2.0], 1)]
        function metric(adv_samples, labels)
            @test length(adv_samples) == length(labels)
            return length(adv_samples)
        end
        result = Interface.benchmark(DummyAttack(), DummyModel(), dataset, metric)
        @test result == 2
    end
end
end # module
