using Test
using AdversarialAttacks

@testset "Attack abstractions" begin
    @test isabstracttype(AbstractAttack)
    @test isabstracttype(WhiteBoxAttack)
    @test isabstracttype(BlackBoxAttack)
    @test WhiteBoxAttack <: AbstractAttack
    @test BlackBoxAttack <: AbstractAttack

    struct DummyWB <: WhiteBoxAttack
        params::Dict{String,Any}
    end

    AdversarialAttacks.name(::DummyWB) = "dummy-wb"
    AdversarialAttacks.hyperparameters(d::DummyWB) = d.params
    AdversarialAttacks.attack(sample, model, ::DummyWB; kwargs...) = (:adv, sample, model, kwargs)

    dummy = DummyWB(Dict("eps" => 0.1))
    @test name(dummy) == "dummy-wb"
    @test hyperparameters(dummy) == Dict("eps" => 0.1)
    adv, sample, model, kwargs = run(:x, :m, dummy; steps=5)
    @test adv == :adv
    @test sample == :x
    @test model == :m
    @test (; kwargs...) == (; steps=5)
end
