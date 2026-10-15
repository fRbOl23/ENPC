using Plots
using Random, Statistics
using JSON

# ---------------------------------------------------------------
# Données (assets/data.json)
# ---------------------------------------------------------------
const data = JSON.parsefile(joinpath(@__DIR__, "assets", "data.json"))
const SI = Dict("g" => 1e-3, "cm" => 1e-2, "mm" => 1e-3, "µm" => 1e-6)

"Valeur d'une donnée du montage (m, L, l, h), convertie en unités SI."
donnee(nom) = (d = data["donnees"][nom]; d["valeur"] * SI[d["unite"]])

"Mesures brutes d'un essai : (nombre de rondelles, distance feuille-table)."
function mesures(id, nom = "Essai 1")
    exp = only(e for e in data["experiences"] if e["id"] == id)
    s = only(s for s in exp["essais"] if s["nom"] == nom)
    return Int.(s["rondelles"]), Float64.(s["distance"])
end

# ---------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------
const g = 9.81                              # m/s²
const M_rondelle = donnee("m")              # kg  masse mesurée d'une rondelle
const uD = SI[data["unites"]["distance"]]   # m   unité des distances mesurées
const uδ = donnee("u_delta")                # m   incertitude de lecture sur δ

Random.seed!(0)   # tirages Monte-Carlo reproductibles : resultats.json ne change qu'avec les données

# Expérience 1 : papier entier (3 essais)
D1, D2, D3 = (mesures(1, "Essai $i") for i in 1:3)

# Expérience 2 : papier 1/3
D4 = mesures(2)

# Expérience 3 : papier 2/3
D5 = mesures(3)

# Expérience 4 : papier 2/3 + épaisseur 2
D6 = mesures(4)

# Expérience 5 : pont en arc (mesures dans le désordre)
D7, D8 = mesures(5, "Essai A"), mesures(5, "Essai B")

# ---------------------------------------------------------------
# Outils
# ---------------------------------------------------------------

"Nombre de rondelles (trié) et déformation (m) d'un essai, référence = 0 rondelle : δ = |Y - Y(0)|."
function essai((X, Y))
    p = sortperm(X)
    X, Y = X[p], Y[p] # Trie en fonction du nombre de rondelles
    δ = abs.(Y .- Y[findfirst(==(0), X)]) .* uD
    return X, δ
end

"Regroupe plusieurs essais (δ calculé essai par essai)."
function regrouper(Ys...)
    paires = [essai(Y) for Y in Ys]
    return reduce(vcat, first.(paires)), reduce(vcat, last.(paires))
end

"Extraction de la raideur k à partir d'une régression linéaire δ = a + b F ; k = 1/b."
function raideur(F, δ, uδ=uδ, N = 10000)
    Δ = δ .+ randn(length(δ), N) .* uδ        # N tirages bruités, un par colonne
    coefs = [ones(length(F)) F] \ Δ           # 2×N : ligne 1 = a, ligne 2 = b
    a, b = coefs[1, :], coefs[2, :]
    # k = 1/b : on propage l'incertitude sur la pente plutôt que de moyenner 1/b,
    # qui diverge quand la pente est mal déterminée (expérience 5)
    k, uk = 1 / mean(b), std(b) / mean(b)^2
    return k, uk, a, b
end

"Trace δ en fonction de F avec la régression linéaire ; raideur k = 1/pente"
function tracer!(fig, X, δ, style; label = "", points = true)
    F = M_rondelle * g .* X
    points && scatter!(fig, F, δ .* 1e3; yerror = uδ * 1e3, label = label, style...)
    k, uk, a, b = raideur(F, δ)
    Fs = range(0, maximum(F), length = 100)
    droites = (a' .+ Fs .* b') .* 1e3         # 100×N : une droite par tirage
    plot!(fig, Fs, vec(mean(droites, dims = 2));
          ribbon = vec(std(droites, dims = 2)), fillalpha = 0.2,
          label = "$label  (régression linéaire, k = $(round(k, digits = 1)) ± $(round(uk, digits = 1)) N/m)",
          color = get(style, :color, :black), linewidth = 2)
    return k, uk
end

function figure_vide(titre)
    return plot(; xlabel = "Force F = m g (N)", ylabel = "Déformation δ (mm)",
                title = titre, legend = :topleft, framestyle = :box)
end

# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------

# Expérience 1 : trois essais et régression sur l'ensemble des points
fig1 = figure_vide("Expérience 1 : papier entier (3 essais)")
for (D, c, nom) in ((D1, :blue, "Essai 1"),
                    (D2, :red, "Essai 2"),
                    (D3, :green, "Essai 3"))
    X, δ = essai(D)
    scatter!(fig1, M_rondelle * g .* X, δ .* 1e3; yerror = uδ * 1e3,
             color = c, marker = :circle, label = nom)
end
k1 = tracer!(fig1, regrouper(D1, D2, D3)..., (color = :black,);
        label = "Essais 1-3", points = false)

# Expériences 2 à 4 : une figure par configuration
fig2 = figure_vide("Expérience 2 : Bande de largeur 1/3 l")
k2 = tracer!(fig2, regrouper(D4)..., (color = :red, marker = :square); label = "Bande de largeur 1/3 l")

fig3 = figure_vide("Expérience 3 : Bande de largeur 2/3 l")
k3 = tracer!(fig3, regrouper(D5)..., (color = :green, marker = :diamond); label = "Bande de largeur 2/3 l")

fig4 = figure_vide("Expérience 4 : Bande de largeur 2/3 l et d'épaisseur 2h")
k4 = tracer!(fig4, regrouper(D6)..., (color = :purple, marker = :utriangle);
        label = "Bande de largeur 2/3 l et d'épaisseur 2h")

# Expérience 5 : pont en arc (données dans le désordre, donc triées par essai())
fig5 = figure_vide("Expérience 5 : Pont en arc")
k5 = tracer!(fig5, regrouper(D7)..., (color = :orange, marker = :circle); label = "Pont en arc")

display(fig1); display(fig2); display(fig3); display(fig4); display(fig5)
for (fig, nom) in ((fig1, "exp1_moyenne"), (fig2, "exp2_un_tiers"), (fig3, "exp3_deux_tiers"),
                   (fig4, "exp4_epaisseur_2"), (fig5, "exp5_cloche"))
    savefig(fig, joinpath(@__DIR__, "assets", "$nom.png"))
end

# ---------------------------------------------------------------
# Résultats (assets/resultats.json, lus par main.typ) — unités SI
# ---------------------------------------------------------------
L, l, h = donnee("L"), donnee("l"), donnee("h")
I = l * h^3 / 12                  # m⁴  moment quadratique de la section
facteur = L^3 / (48 * I)          # E = k L³ / (48 I)

resultats = Dict(
    "unites" => Dict("I" => "m^4", "k" => "N/m", "E" => "Pa"),
    "I" => I,
    "k" => Dict(id => Dict("valeur" => k, "incertitude" => uk)
                for (id, (k, uk)) in Dict("1" => k1, "2" => k2, "3" => k3, "4" => k4, "5" => k5)),
    "E" => Dict("valeur" => k1[1] * facteur, "incertitude" => k1[2] * facteur),
)
open(joinpath(@__DIR__, "assets", "resultats.json"), "w") do io
    JSON.json(io, resultats; pretty = true)
end