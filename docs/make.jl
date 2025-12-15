using AdversarialAttacks
using Documenter

DocMeta.setdocmeta!(AdversarialAttacks, :DocTestSetup, :(using AdversarialAttacks); recursive=true)

makedocs(;
    modules=[AdversarialAttacks],
    authors="FirstName LastName <orestis.papandreou@campus.tu-berlin.de>",
    sitename="AdversarialAttacks.jl",
    format=Documenter.HTML(;
        canonical="https://shnaky.github.io/AdversarialAttacks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Attack Interface" => "attack_interface.md",
<<<<<<< HEAD
        "Black Box Algorithms" => "blackbox_subtypes.md"
=======
        "Model Interface" => "model_interface.md",
>>>>>>> origin/main
    ],
)

deploydocs(;
    repo="github.com/shnaky/AdversarialAttacks.jl",
    devbranch="main",
)
