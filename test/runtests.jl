using AdversarialAttacks
using Test

@testset "AdversarialAttacks.jl" begin
    include("models/FgsmModel.jl")
    include("attacks/FgsmAttack.jl")
end
