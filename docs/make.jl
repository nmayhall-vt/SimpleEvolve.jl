using SimpleEvolve
using Documenter

DocMeta.setdocmeta!(SimpleEvolve, :DocTestSetup, :(using SimpleEvolve); recursive=true)

makedocs(;
    modules=[SimpleEvolve],
    authors="Nick and Arnab",
    sitename="SimpleEvolve.jl",
    format=Documenter.HTML(;
        canonical="https://nmayhall-vt.github.io/SimpleEvolve.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nmayhall-vt/SimpleEvolve.jl",
    devbranch="main",
)
