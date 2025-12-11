using AdversarialAttacks
using Documenter

DocMeta.setdocmeta!(AdversarialAttacks, :DocTestSetup, :(using AdversarialAttacks); recursive=true)

makedocs(;
    modules=[AdversarialAttacks, AdversarialAttacks.Attack, AdversarialAttacks.FastGradientSignMethod],
    authors="FirstName LastName <orestis.papandreou@campus.tu-berlin.de>",
    sitename="AdversarialAttacks.jl",
    format=Documenter.HTML(;
        canonical="https://shnaky.github.io/AdversarialAttacks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Fast Gradient Sign Method Attack" => "fgsm.md",
    ],
)

deploydocs(;
    repo="github.com/shnaky/AdversarialAttacks.jl",
    devbranch="main",
)
