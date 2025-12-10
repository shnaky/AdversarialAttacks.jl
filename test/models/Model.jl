using Test
using AdversarialAttacks

const Model = AdversarialAttacks.Model

@testset "Model abstractions" begin

  @test isabstracttype(Model.AbstractModel)
  @test isabstracttype(Model.DifferentiableModel)
  @test isabstracttype(Model.NonDifferentiableModel)
  @test Model.DifferentiableModel <: Model.AbstractModel
  @test Model.NonDifferentiableModel <: Model.AbstractModel

  struct DummyModel <: Model.AbstractModel
    factor::Int
  end

  Model.name(::DummyModel) = "DummyModel"
  Model.predict(m::DummyModel, x) = m.factor * x
  Model.loss(m::DummyModel, x, y) = sum(abs.(Model.predict(m, x) .- y))
  Model.params(m::DummyModel) = (m.factor,)

  m = DummyModel(2)

  @test Model.name(m) == "DummyModel"
  @test Model.predict(m, [1, 2, 3]) == [2, 4, 6]
  @test Model.loss(m, [1, 2], [2, 4]) == 0.0
  @test Model.params(m) == (2,)
end