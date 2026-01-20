using Test
using AdversarialAttacks

@testset "Attack abstractions" begin
    @testset "Abstract types" begin
        @test isabstracttype(AbstractAttack)
        @test isabstracttype(WhiteBoxAttack)
        @test isabstracttype(BlackBoxAttack)
        @test WhiteBoxAttack <: AbstractAttack
        @test BlackBoxAttack <: AbstractAttack
    end

    @testset "Default name() implementation" begin
        struct TestAttackDefaultName <: AbstractAttack end

        m = TestAttackDefaultName()

        @test AdversarialAttacks.name(m) == "TestAttackDefaultName"
        @test AdversarialAttacks.name(m) == string(typeof(m))
    end

    @testset "Default hyperparameters() implementation" begin
        struct TestAttackDefaultHyperparams <: AbstractAttack end

        atk = TestAttackDefaultHyperparams()

        @test AdversarialAttacks.hyperparameters(atk) == Dict{String,Any}()
        @test isempty(AdversarialAttacks.hyperparameters(atk))
    end

    @testset "DummyAttack implementation" begin
        struct DummyAttack <: AbstractAttack
            params::Dict{String,Any}
        end

        AdversarialAttacks.name(::DummyAttack) = "DummyAttack"
        AdversarialAttacks.hyperparameters(d::DummyAttack) = d.params
        AdversarialAttacks.craft(sample, model, ::DummyAttack; kwargs...) = (:adv, sample, model, kwargs)

        dummy = DummyAttack(Dict("eps" => 0.1))
        @test name(dummy) == "DummyAttack"
        @test hyperparameters(dummy) == Dict("eps" => 0.1)
        adv, sample, model, kwargs = craft(:x, :m, dummy; steps=5)
        @test adv == :adv
        @test sample == :x
        @test model == :m
        @test (; kwargs...) == (; steps=5)
    end

    @testset "craft fallback MethodError" begin
        sample = (data=[1.0], label=1)
        struct MockModel <: AbstractModel end
        struct MockAttack <: AbstractAttack end

        # fallback dispatch hit
        @test_throws MethodError craft(sample, MockModel(), MockAttack())
        @test_throws MethodError attack(MockAttack(), MockModel(), sample)
    end
end
