# Construit le compte-rendu : data.json → main.jl → resultats.json + figures → main.pdf
# Usage : julia build.jl
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()            # installe les dépendances au premier lancement (Manifest.toml)

include(joinpath(@__DIR__, "main.jl"))
run(`typst compile $(joinpath(@__DIR__, "main.typ"))`)
println("main.pdf à jour")
