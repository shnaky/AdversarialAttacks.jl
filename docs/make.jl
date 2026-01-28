using AdversarialAttacks
using Documenter
using Literate

# Create index.md from README
cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force = true)

DocMeta.setdocmeta!(
    AdversarialAttacks,
    :DocTestSetup,
    :(using AdversarialAttacks);
    recursive = true,
)

# --- Literate.jl: convert example scripts to markdown tutorials ---
EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
TUTORIALS_OUTPUT = joinpath(@__DIR__, "src", "tutorials")

for filename in [
    "whitebox_fgsm_flux_mnist.jl",
    "blackbox_basicrandomsearch_decisiontree_iris.jl",
]
    Literate.markdown(
        joinpath(EXAMPLES_DIR, filename),
        TUTORIALS_OUTPUT;
        documenter = true,
        credit = false,
    )
end

makedocs(;
    modules = [AdversarialAttacks],
    authors = "FirstName LastName <orestis.papandreou@campus.tu-berlin.de>",
    sitename = "AdversarialAttacks.jl",
    format = Documenter.HTML(;
        canonical = "https://shnaky.github.io/AdversarialAttacks.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Getting Started" => "index.md",
        "Tutorials & Examples" => [
            "Overview" => "examples/index.md",
            "White-Box – FGSM (Flux, MNIST)" => "tutorials/whitebox_fgsm_flux_mnist.md",
            "Black-Box – Basic Random Search (DecisionTree, Iris)" => "tutorials/blackbox_basicrandomsearch_decisiontree_iris.md",
        ],
        "Developer Documentation" => [
            "Attack Interface" => "attack_interface.md",
            "FGSM (White-Box)" => "FGSM.md",
            "BasicRandomSearch (Black-Box)" => "BasicRandomSearch.md",
            "Robustness Evaluation Suite" => "evaluation.md",
            "Interface" => "interface.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/shnaky/AdversarialAttacks.jl",
    devbranch = "main",
)
