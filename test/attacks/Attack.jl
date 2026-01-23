using Test
using AdversarialAttacks
using Flux

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

    @testset "DummyAttack implementation" begin
        struct DummyAttack <: AbstractAttack end

        AdversarialAttacks.name(::DummyAttack) = "DummyAttack"
        AdversarialAttacks.attack(::DummyAttack, model, sample; kwargs...) = (:adv, model, sample, kwargs)

        dummy = DummyAttack()
        @test name(dummy) == "DummyAttack"
        adv, model, sample, kwargs = attack(dummy, :m, :x; steps = 5)
        @test adv == :adv
        @test model == :m
        @test sample == :x
        @test (; kwargs...) == (; steps = 5)
    end

    @testset "attack fallback MethodError" begin
        sample = (data = [1.0], label = 1)
        struct MockModel end
        struct MockAttack <: AbstractAttack end

        # fallback dispatch hit
        @test_throws MethodError attack(MockAttack(), MockModel(), sample)
    end
end
