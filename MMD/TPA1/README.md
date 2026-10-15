# TP A1 — Ponts en papier

Compte-rendu du TP A1 de Mécanique des milieux déformables (ENPC) : flexion d'une bande de papier (maquette 1) et pont en arc (maquette 2).

## Prérequis

- [Julia](https://julialang.org/downloads/) (≥ 1.13, version du `Manifest.toml`)
- [Typst](https://github.com/typst/typst#installation) (≥ 0.14)

## Compiler le compte-rendu

```sh
julia build.jl
```

Au premier lancement, les paquets Julia (versions figées dans `Manifest.toml`) sont installés automatiquement, ce qui peut prendre quelques minutes.

Pour ne modifier que le texte, sans recalculer :

```sh
typst watch main.typ
```

## Chaîne de compilation

```
assets/data.json ──▶ main.jl ──▶ assets/resultats.json
        │                   └──▶ assets/exp*.png
        ▼                              │
     main.typ ◀────────────────────────┘
        │
        ▼
     main.pdf
```

Aucune valeur numérique n'est écrite en dur dans `main.typ` : tout vient de `data.json` (mesures, dimensions) ou de `resultats.json` (raideurs, module d'Young, calculés par `main.jl`).

| Fichier | Rôle |
|---|---|
| `assets/data.json` | Mesures brutes et dimensions du montage. **Seul fichier de données à modifier à la main.** |
| `main.jl` | Régressions, calcul de `I`, `E` et des raideurs ; écrit `resultats.json` et les figures. |
| `assets/resultats.json` | Résultats calculés (unités SI). Généré : ne pas modifier. |
| `assets/exp*.png` | Figures des régressions. Générées : ne pas modifier. |
| `main.typ`, `template.typ` | Compte-rendu et mise en page. |
| `assets/*.png`, `assets/*.svg` (autres) | Photos et schémas des maquettes. |
| `build.jl` | Lance toute la chaîne. |
| `Project.toml`, `Manifest.toml` | Dépendances Julia. |

## Modifier les données

Dans `assets/data.json` :

- `experiences` : une entrée par expérience (`id`, `configuration`), chacune avec ses `essais` ; un essai associe la liste des `rondelles` (nombre de rondelles posées) aux `distance`s mesurées, dans l'unité donnée par `unites.distance` (cm).
- `donnees` : dimensions et constantes du montage, sous la forme `{ "valeur", "unite", "description" }`. Les unités reconnues sont `g`, `cm`, `mm` et `µm`.

Puis relancer `julia build.jl`.
