using AdversarialAttacks
using Documenter

# Create index.md from README
cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

DocMeta.setdocmeta!(
    AdversarialAttacks,
    :DocTestSetup,
    :(using AdversarialAttacks);
    recursive=true,
)

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
        "Getting Started" => "index.md",
        "Tutorials & Examples" => [
            "Overview" => "examples/index.md",
            "White-Box – FGSM (Flux, MNIST)" => "examples/whitebox_fgsm_flux_mnist.md",
            "Black-Box – Basic Random Search (DecisionTree, Iris)" => "examples/blackbox_basicrandomsearch_decisiontree_iris.md",
        ],
        "Developer Documentation" => [
            "Home" => "index.md",
            "Attack Interface" => "attack_interface.md",
            "Model Interface" => "model_interface.md",
            "Fast Gradient Sign Method Attack" => "fgsm.md",
            "Black Box Attacks" => "blackbox_subtypes.md",
            "Robustness Evaluation Suite" => "evaluation.md",
            "Interface" => "interface.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/shnaky/AdversarialAttacks.jl",
    devbranch="main",
)
