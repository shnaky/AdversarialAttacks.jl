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
  # TODO: change author info
  authors="FirstName LastName <orestis.papandreou@campus.tu-berlin.de>",
  sitename="AdversarialAttacks.jl",
  format=Documenter.HTML(;
    canonical="https://shnaky.github.io/AdversarialAttacks.jl",
    edit_link="main",
    assets=String[], # TODO: create icon
  ),
  pages=[
    "Getting Started" => "index.md",
    "Developer Documentation" => [
      "Attacks" => "attack_interface.md",
      "Model Interface" => "model_interface.md",
    ],
  ],
)

deploydocs(;
  repo="github.com/shnaky/AdversarialAttacks.jl",
  devbranch="main",
)
