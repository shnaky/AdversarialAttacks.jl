using Test
using AdversarialAttacks
using Flux
using Random

@testset "FGSM Struct" begin

    # Test default constructor
    atk = FGSM()
    @test atk isa FGSM
    @test atk.epsilon == 0.1

    # Test constructor with keyword argument
    atk_with_epsilon = FGSM(epsilon = 0.25)
    @test atk_with_epsilon isa FGSM
    @test atk_with_epsilon.epsilon == 0.25

    # Test type hierarchy
    @test FGSM <: WhiteBoxAttack
    @test FGSM <: AbstractAttack

    sample = (data = Float32[1.0, 2.0, 3.0], label = Flux.onehot(1, 1:2))
    model = Chain(
        Dense(3 => 2),
        softmax,
    )

    result = attack(atk_with_epsilon, model, sample)
    @test result isa Vector
    @test size(result) == size(sample.data)
    @test eltype(result) == eltype(sample.data)

end

@testset "FGSM returns detailed result as namedtuple" begin
    Random.seed!(1234)

    # Simple 2-class Flux model
    model = Chain(
        Dense(4 => 2),
        softmax,
    )

    x = Float32[1, 2, 3, 4]
    y = Flux.onehot(1, 1:2)               # true label = 1
    sample = (data = x, label = y)

    atk = FGSM(epsilon = 0.1f0)           # detailed_result default = false

    # 1) Default call: detailed_result=false return adversarial example only
    x_adv_simple = attack(atk, model, sample)
    @test x_adv_simple isa AbstractArray
    @test size(x_adv_simple) == size(x)
    @test x_adv_simple != x               # perturbation should change input

    # 2) Explicit detailed_result=false behaves the same
    x_adv_false = attack(atk, model, sample; detailed_result = false)
    @test x_adv_false isa AbstractArray
    @test x_adv_false == x_adv_simple

    # 3) detailed_result=true return NamedTuple with metadata
    result = attack(atk, model, sample; detailed_result = true)
    @test result isa NamedTuple
    @test haskey(result, :x_adv)
    @test haskey(result, :queries_used)

    @test result.x_adv isa AbstractArray
    @test size(result.x_adv) == size(x)

    # Adversarial example must match the simple-return variant
    @test result.x_adv == x_adv_simple

    # FGSM uses exactly one gradient evaluation
    @test result.queries_used == 1
end
