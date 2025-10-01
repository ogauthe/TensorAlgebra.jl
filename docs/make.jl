using TensorAlgebra: TensorAlgebra
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(TensorAlgebra, :DocTestSetup, :(using TensorAlgebra); recursive = true)

include("make_index.jl")

makedocs(;
    modules = [TensorAlgebra],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "TensorAlgebra.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/TensorAlgebra.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"],
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
    repo = "github.com/ITensor/TensorAlgebra.jl", devbranch = "main", push_preview = true
)
