using Test
using AdversarialAttacks

@testset "Model abstractions" begin
    @testset "Abstract types" begin
        @test isabstracttype(AbstractModel)
        @test isabstracttype(DifferentiableModel)
        @test isabstracttype(NonDifferentiableModel)
        @test DifferentiableModel <: AbstractModel
        @test NonDifferentiableModel <: AbstractModel
    end

    @testset "Default name() implementation" begin
        struct TestModelDefaultName <: AbstractModel end

        AdversarialAttacks.predict(::TestModelDefaultName, x) = x
        AdversarialAttacks.loss(::TestModelDefaultName, x, y) = 0.0
        AdversarialAttacks.params(::TestModelDefaultName) = ()

        m = TestModelDefaultName()

        @test AdversarialAttacks.name(m) == "TestModelDefaultName"
        @test AdversarialAttacks.name(m) == string(typeof(m))
    end

    @testset "DummyModel implementation" begin
        struct DummyModel <: AbstractModel
            factor::Int
        end

        AdversarialAttacks.name(::DummyModel) = "DummyModel"
        AdversarialAttacks.predict(m::DummyModel, x) = m.factor * x
        AdversarialAttacks.loss(m::DummyModel, x, y) = sum(abs.(predict(m, x) .- y))
        AdversarialAttacks.params(m::DummyModel) = (m.factor,)

        m = DummyModel(2)

        @test name(m) == "DummyModel"
        @test predict(m, [1, 2, 3]) == [2, 4, 6]
        @test loss(m, [1, 2], [2, 4]) == 0.0
        @test params(m) == (2,)
    end
end
