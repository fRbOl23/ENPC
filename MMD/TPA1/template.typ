// Gabarit de compte-rendu de TP (ENPC — MMD)

#let accent = rgb("#1f4e79")

// Encadré pour un résultat important
#let resultat(body) = align(center, block(
  fill: accent.lighten(90%),
  stroke: (left: 2pt + accent),
  inset: (x: 14pt, y: 10pt),
  radius: 2pt,
  body,
))

// Remarque / note en marge du raisonnement
#let remarque(body) = block(
  inset: (left: 10pt, y: 4pt),
  stroke: (left: 1pt + luma(170)),
  text(fill: luma(70), size: 0.95em)[_Remarque._ #body],
)

// Marqueur visible de travail restant
#let todo(body) = box(
  fill: rgb("#fff3cd"),
  stroke: 0.6pt + rgb("#e0a800"),
  inset: (x: 4pt, y: 2pt),
  radius: 2pt,
  text(size: 0.85em, fill: rgb("#7a5a00"))[*À faire :* #body],
)

// Tableau à filets horizontaux (style « booktabs »)
#let tableau(..args) = table(
  stroke: none,
  align: center + horizon,
  inset: (x: 6pt, y: 5pt),
  ..args,
)

#let rapport(
  titre: none,
  sous-titre: none,
  cours: none,
  etablissement: none,
  auteurs: (),
  date: datetime.today(),
  sommaire: true,
  body,
) = {
  set document(title: titre, author: auteurs)
  set text(lang: "fr", size: 11pt)
  set par(justify: true, leading: 0.65em, spacing: 1.1em)

  set page(
    paper: "a4",
    margin: (x: 2.3cm, top: 2.6cm, bottom: 2.4cm),
    header: context {
      if counter(page).get().first() > 1 {
        set text(size: 9pt, fill: luma(100))
        grid(
          columns: (1fr, 1fr),
          align(left, cours), align(right, titre),
        )
        v(-6pt)
        line(length: 100%, stroke: 0.4pt + luma(170))
      }
    },
    footer: context align(center, text(size: 9pt, fill: luma(100))[
      #counter(page).display("1 / 1", both: true)
    ]),
  )

  // Titres
  set heading(numbering: "1.1")
  show heading: set text(fill: accent)
  show heading.where(level: 1): it => {
    v(0.8em)
    set text(size: 15pt)
    it
    v(0.2em)
  }
  show heading.where(level: 2): set text(size: 12.5pt)
  show heading.where(level: 3): set text(size: 11pt)

  // Figures et tableaux
  set figure(gap: 0.8em)
  show figure.caption: set text(size: 9.5pt)
  show figure.caption: it => [*#it.supplement #context it.counter.display(it.numbering)* : #it.body]
  show figure.where(kind: table): set figure.caption(position: top)
  show figure: set block(spacing: 1.6em)

  // Listes et équations
  set list(indent: 0.8em, marker: ([•], [–]))
  set enum(indent: 0.8em)
  show math.equation.where(block: true): set block(spacing: 1em)

  // Page de titre
  align(center)[
    #v(0.5cm)
    #text(size: 10pt, fill: luma(90), upper(etablissement))
    #v(0.1cm)
    #text(size: 10pt, fill: luma(90), cours)
    #v(0.6cm)
    #line(length: 60%, stroke: 0.8pt + accent)
    #v(0.2cm)
    #text(size: 22pt, weight: "bold", fill: accent, titre)
    #if sous-titre != none {
      v(0.1cm)
      text(size: 13pt, style: "italic", sous-titre)
    }
    #v(0.2cm)
    #line(length: 60%, stroke: 0.8pt + accent)
    #v(0.4cm)
    #auteurs.join(" · ")
    #v(0.1cm)
    #text(size: 10pt, fill: luma(90), date.display("[day]/[month]/[year]"))
  ]

  if sommaire {
    v(0.6cm)
    block(
      width: 100%,
      inset: 12pt,
      fill: luma(247),
      radius: 3pt,
      outline(title: [Sommaire], depth: 2, indent: auto),
    )
  }
  v(0.6cm)

  body
}
