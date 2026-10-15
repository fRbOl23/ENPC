#import "template.typ": *

// Données expérimentales (assets/data.json)
#let data = json("assets/data.json")
#let nb(x) = if type(x) == float and calc.fract(x) == 0 { str(x) + ".0" } else { str(x) }
#let val(nom) = { let d = data.donnees.at(nom); $#nb(d.valeur) #d.unite$ }

// Résultats calculés par main.jl (assets/resultats.json, unités SI)
#let res = json("assets/resultats.json")
#let k(id) = res.k.at(str(id))

// Raideurs prédites par le modèle à partir de k (expérience 1), avec k ∝ l h³
#let facteur = ("2": 1 / 3, "3": 2 / 3, "4": 8 * 2 / 3, "5": 2096 / 48)  // "5" : pont en arc, k_tot = 2096 E I / L³
#let modele(id) = {
  let f = facteur.at(str(id))
  (valeur: f * k(1).valeur, incertitude: f * k(1).incertitude)
}

// Mise en forme : x à d décimales, zéros finaux compris (d < 0 : arrondi aux dizaines, ...)
#let fixe(x, d) = {
  if d <= 0 { return str(int(calc.round(x / calc.pow(10, -d)) * calc.pow(10, -d))) }
  let s = str(calc.round(x, digits: d))
  let (e, f) = if s.contains(".") { s.split(".") } else { (s, "") }
  e + "." + f + "0" * (d - f.len())
}
// Décimales correspondant au premier chiffre significatif de l'incertitude u
#let decimales(u) = -int(calc.floor(calc.log(u)))
// Valeur, incertitude, et « valeur ± incertitude » d'un résultat r, multipliés par f
#let arrondi(r, f: 1) = fixe(r.valeur * f, decimales(r.incertitude * f))
#let inc(r, f: 1) = fixe(r.incertitude * f, decimales(r.incertitude * f))
#let pm(r, f: 1, unite: none) = $(#arrondi(r, f: f) plus.minus #inc(r, f: f)) #unite$
// Notation scientifique à n chiffres significatifs
#let sci(x, n: 3) = {
  let e = int(calc.floor(calc.log(x)))
  $#fixe(x / calc.pow(10.0, e), n - 1) dot 10^(#e)$
}

#show: rapport.with(
  titre: [TPA1 : Ponts en papier],
  sous-titre: [Étude expérimentale de la flexion d'une bande et d'un pont en arc],
  cours: [Mécanique des milieux déformables],
  etablissement: [École nationale des ponts et chaussées],
  auteurs: ("BOURHLAM Sara", "CHAMINAS Stella", "CHEN Wanli", "CASALS--IRAHOLA Joaquim"),
)

// Tableau des mesures d'une expérience dont les essais partagent les mêmes rondelles,
// découpé en blocs d'au plus `max` colonnes
#let tableau-mesures(id, max: 13) = {
  let e = data.experiences.find(e => e.id == id)
  let x = e.essais.first().rondelles
  let n = calc.min(max, x.len())
  let titre(s) = if e.essais.len() == 1 [Distance] else [#s.nom]
  let blocs = range(0, x.len(), step: n).map(i => {
    let j = calc.min(i + n, x.len())
    let vide = ([],) * (n - (j - i))
    (
      table.hline(stroke: if i == 0 { 0.8pt } else { 0.5pt }),
      [*Rondelles*], ..x.slice(i, j).map(r => [*#r*]), ..vide,
      table.hline(stroke: 0.5pt),
      ..e.essais.map(s => (
        [#titre(s) (#data.unites.distance)], ..s.distance.slice(i, j).map(d => [#nb(d)]), ..vide,
      )).flatten(),
    )
  })
  tableau(columns: (auto,) + (1fr,) * n, ..blocs.flatten(), table.hline(stroke: 0.8pt))
}

// Maquette 1

= Maquette 1 : flexion de la bande

Une première approche du comportement élastique d'un matériau soumis à une force consiste à observer expérimentalement sa déformation face à une action extérieure. Dans ce travail, nous avons étudié la manière dont le papier aquarelle se déforme lorsqu'il subit le poids d'un certain nombre de rondelles en métal (cf. @fig:experience).

== Protocole expérimental

=== Conception du montage

+ Découper une bande de papier aquarelle de longueur $L_"tot" = #val("L_tot")$, de largeur $l = #val("l")$ et d'épaisseur $h = #val("h")$.
+ Concevoir deux rouleaux compacts de papier de même largeur $l$.
+ Coller à l'aide de colle forte chaque rouleau à $5 "cm"$ des bords de la bande.
+ Surélever le système ainsi construit et marquer le centre de la bande.

#figure(
  image("assets/ponts.png", width: 70%),
  caption: [Photo du montage réalisé.],
) <fig:montage>

=== Mesure

Placer progressivement des rondelles de métal sur le marquage et mesurer le déplacement vertical du milieu du pont.

#figure(
  image("assets/experience1.png", width: 70%),
  caption: [Photo de l'expérience.],
) <fig:experience>

== Mesures expérimentales

La hauteur du milieu de la bande par rapport à la table est relevée après l'ajout de chaque rondelle (@tab:mesures). La masse d'une rondelle est $m = #val("m")$.

#figure(
  tableau-mesures(1),
  caption: [Hauteur du milieu de la bande en fonction du nombre de rondelles.],
) <tab:mesures>

#figure(
  image("assets/exp1_moyenne.png", width: 75%),
  caption: [Déformation $delta$ en fonction de la force appliquée $F = m g$ (trois essais).],
) <fig:exp1>

== Modélisation mécanique

=== Hypothèses

- *Petites déformations* : les déformations et les déplacements sont très petits devant la longueur du pont. Cela n'a été que partiellement vérifié expérimentalement : pour un grand nombre de rondelles, la déformation devenait presque du même ordre de grandeur que la longueur du pont.
- *Navier--Bernoulli* : la portée des éléments de structure est très grande devant leur hauteur. C'est vérifié, l'épaisseur du papier étant très petite devant la longueur du pont.
- Le poids du papier est négligé devant celui des rondelles métalliques.
- Le papier reste dans son domaine élastique.

=== Modèle

#figure(
  image("assets/model.svg", width: 75%),
  caption: [Modèle mécanique de la bande soumise à une force.]
)

=== Équations de la statique

On applique le principe fondamental de la statique à l'ensemble du système, à l'équilibre :
$ R_A arrow(e)_y + R_B arrow(e)_y - F arrow(e)_y = arrow(0) quad ==> quad R_A = R_B = F / 2. $

Soit $x$ l'abscisse curviligne du point étudié. Les efforts intérieurs s'écrivent :

$
  x > L/2 : & quad N = 0, quad V = R_B, && quad M = R_B (L - x) \
  x < L/2 : & quad N = 0, quad V = R_B - F, && quad M = R_B (L - x) - F (L / 2 - x)
$

=== Équations de déformation

Les lois de comportement donnent
$ epsilon = N / (E A) = xi'_t, wide chi = M / (E I) = xi''_n. $

On intègre ces deux équations. D'une part, $xi_t$ est constant ; or $xi_t (0) = 0$, donc $xi_t = 0$. D'autre part :
$
  xi'_n (x) = cases(
    -F / (4 E I) (L - x)^2 + A_1 & "si" x > L/2,
    -F / (4 E I) (L - x)^2 + F / (2 E I) (L / 2 - x)^2 + A_2 & "si" x < L/2,
  )
$

En intégrant une seconde fois :
$
  xi_n (x) = cases(
    F / (12 E I) (L - x)^3 + A_1 x + B_1 & "si" x > L/2,
    F / (12 E I) (L - x)^3 - F / (6 E I) (L / 2 - x)^3 + A_2 x + B_2 & "si" x < L/2,
  )
$

Les conditions aux limites sont
$ xi_n (0) = 0, quad xi_n (L) = 0, quad xi'_n (L/2) = 0 quad "et" xi'_n "continue en" L/2, $
ce qui donne
$ A_1 = A_2 = (F L^2) / (16 E I), wide B_1 = B_2 = -(F L^3) / (16 E I). $

On cherche la flèche au centre, $xi_n (L/2) = f(F)$ :
$ xi_n (L / 2) = -(L^3) / (48 E I) F quad ==> quad delta = (L^3) / (48 E I) F. $

=== Détermination du module d'Young

Le moment quadratique de la section vaut $I = l h^3 / 12 = #sci(res.I) "m"^4$. La régression linéaire (@fig:exp1) donne $F = k delta$ avec
$ k = (48 E I) / L^3 = #arrondi(k(1)) "N/m", $
d'où le module d'Young
$ E = (k L^3) / (48 I) = #arrondi(res.E, f: 1e-6) "MPa". $

*Incertitude.* La régression linéaire, en prenant une incertitude de $#val("u_delta")$ sur $delta$, donne $u(k) = #inc(k(1)) "N/m"$. Par ailleurs, $I$ a été mesuré expérimentalement et son incertitude relative est très faible devant celle de $k$. Ainsi
$ u(E) = u(k) L^3 / (48 I) = #inc(res.E, f: 1e-6) "MPa". $

#resultat[$ E = #pm(res.E, f: 1e-6, unite: "MPa") $]

Après avoir mis en évidence la relation linéaire entre la force appliquée au papier et le déplacement du centre, il est à présent intéressant de faire varier plusieurs paramètres, tels que l'épaisseur et la largeur de la bande, et d'étudier le nouveau comportement du système.

== Influence des dimensions de la bande

D'après la modélisation précédente, la raideur du pont s'écrit
$ k = (48 E I) / L^3 = (48 E) / L^3 dot (l h^3) / 12 = (4 E l h^3) / L^3. $

*Protocole.* Marquer dans le sens de la largeur la bande de papier construite au début du travail en trois parts égales. Découper la bande dans le sens de la longueur au niveau d'une des marques pour obtenir deux bandes de largeurs $l/3$ et $2l/3$.

=== Bande de largeur $l/3$

On s'intéresse dans un premier temps à la bande de largeur $l/3$ et on réitère l'expérience avec les rondelles (@tab:exp2).

#figure(
  image("assets/experience2.png", width: 70%),
  caption: [Photo de l'expérience pour la bande de largeur $l/3$.],
) <fig:experience-tiers>

#figure(
  tableau-mesures(2),
  caption: [Hauteur du milieu de la bande de largeur $l/3$ en fonction du nombre de rondelles.],
) <tab:exp2>

#figure(
  image("assets/exp2_un_tiers.png", width: 75%),
  caption: [Déformation en fonction de la force pour la bande de largeur $l/3$.],
) <fig:exp2>

*Modélisation.* La raideur étant proportionnelle à la largeur,
$ k_(1/3) = (4 E h^3) / L^3 dot l / 3 = k / 3 = #arrondi(modele(2)) "N/m". $
D'autre part, la régression donne $k_(1/3) = #pm(k(2), unite: "N/m")$.

#remarque[On remarque une non linéarité pour des valeurs de δ élevées (@fig:exp2). Cela est peut-être dû au non-respect de l’hypothèse des petites perturbations. En effet, lorsque δ augmente, δ se rapproche de l’ordre de grandeur de L et l’hypothèse des petites perturbations n’est plus vérifiée.]

=== Bande de largeur $2l/3$

On s'intéresse à présent à la bande de largeur $2l/3$ (@tab:exp3).

#figure(
  tableau-mesures(3),
  caption: [Hauteur du milieu de la bande de largeur $2l/3$ en fonction du nombre de rondelles.],
) <tab:exp3>

#figure(
  image("assets/exp3_deux_tiers.png", width: 75%),
  caption: [Déformation en fonction de la force pour la bande de largeur $2l/3$.],
) <fig:exp3>

*Modélisation.* De même,
$ k_(2/3) = (4 E h^3) / L^3 dot (2 l) / 3 = 2 / 3 k = #arrondi(modele(3)) "N/m". $
D'autre part, la régression donne $k_(2/3) = #pm(k(3), unite: "N/m")$ : le modèle est bien vérifié.

=== Bande d'épaisseur $2h$

Il semble maintenant pertinent d'étudier comment l'épaisseur du pont influe sur la déformation.

*Protocole.* Dans une autre feuille de papier aquarelle, couper une bande de mêmes dimensions que la bande $2l/3$, puis la coller par-dessus. On obtient une nouvelle bande de largeur $2l/3$ et d'épaisseur $2h$, dont les mesures sont reportées dans le @tab:exp4.

#figure(
  tableau-mesures(4),
  caption: [Hauteur du milieu de la bande de largeur $2l/3$ et d'épaisseur $2h$ en fonction du nombre de rondelles.],
) <tab:exp4>

#figure(
  image("assets/exp4_epaisseur_2.png", width: 75%),
  caption: [Déformation en fonction de la force pour la bande de largeur $2l/3$ et d'épaisseur $2h$.],
) <fig:exp4>

*Modélisation.* On avait $k_(2/3) = (8 E l h^3) / (3 L^3)$. Ainsi
$ k_(2/3, "épais") = (8 E (2h)^3 l) / (3 L^3) = 8 dot (8 E h^3 l) / (3 L^3) = 8 k_(2/3) = #arrondi(modele(4)) "N/m". $
D'autre part, la régression donne $k_(2/3, "épais") = #pm(k(4), unite: "N/m")$.

#remarque[On constate un écart de $#calc.round((k(4).valeur / modele(4).valeur - 1) * 100) %$, imputable au montage : la colle utilisée peut être à l'origine d'une raideur supérieure à celle attendue.]

== Bilan

Pour conclure, on remarque que le modèle mécanique que nous avons choisi pour le pont représente bien la réalité, à l'exception d’une légère non-linéarité pour l’expérience avec la plus faible largeur. On suppose que cet écart au modèle est dû à l’hypothèse des petites perturbations qui n’est plus respecté pour de grands déplacements du tablier. 


#pagebreak()

// Maquette 2

= Maquette 2 : pont en arc

== Réalisation expérimentale

=== Protocole expérimental

+ Découper deux bandes de papier aquarelle de longueur $L_"tot" = #val("L_tot")$ et de largeur $l = #val("l")$.
+ Concevoir deux rouleaux compacts de même rayon et de hauteur $l$ pour les supports du pont.
+ Concevoir deux rouleaux compacts de même rayon et de hauteur $h_p = #val("h_p")$ pour les piliers.
+ Coller les extrémités des deux bandes de papier l'une sur l'autre de façon à former un arc de hauteur $h_p$.
+ Coller les piliers bien centrés au niveau de la flèche de l'arche.
+ Coller les rouleaux supports du pont.

#figure(
  tableau(
    columns: (auto, auto, auto),
    align: (left, center, center),
    table.hline(stroke: 0.8pt),
    table.header([*Grandeur*], [*Symbole*], [*Valeur*]),
    table.hline(stroke: 0.5pt),
    [Largeur du pont], $l$, val("l"),
    [Longueur du pont (entre les deux appuis)], $L$, val("L"),
    [Longueur totale du pont], $L_"tot"$, val("L_tot"),
    [Hauteur du pilier central], $h_p$, val("h_p"),
    [Épaisseur du papier], $h$, val("h"),
    table.hline(stroke: 0.8pt),
  ),
  caption: [Dimensions du pont en arc.],
) <tab:dimensions-arc>

Nous avons choisi de construire le pont en collant les deux extrémités de l'arche de sorte que la fin de la courbe de l'arche soit sur les appuis. Le pilier central est constitué de deux rouleaux de papier, chacun collé à la verticale sur un bord du tablier.

#figure(
  image("assets/ponts_arc_legende.png"),
  caption: [Photo légendée du pont en arc]
)

#figure(
  image("assets/ponts_arc_haut.png"),
  caption: [Photo du pont en arc vue d'en haut]
)

=== Protocole de mesure

Nous avons ajouté des rondelles par paquets de 5 et mesuré la distance entre la table et le centre du tablier.

== Mesures expérimentales

#let exp5 = data.experiences.find(e => e.id == 5)
#let n5 = calc.max(..exp5.essais.map(e => e.rondelles.len()))
#let ligne(titre, xs) = (titre, ..xs.map(x => [#nb(x)]), ..([],) * (n5 - xs.len()))
#figure(
  tableau(
    columns: (auto,) + (1fr,) * n5,
    table.hline(stroke: 0.8pt),
    ..exp5.essais
      .map(e => ligne([*#e.nom* : rondelles], e.rondelles) + ligne([Distance (#data.unites.distance)], e.distance))
      .join((table.hline(stroke: 0.5pt),)),
    table.hline(stroke: 0.8pt),
  ),
  caption: [Hauteur du centre du tablier en fonction du nombre de rondelles (pont en arc), dans l'ordre des mesures.],
) <tab:mesures-arc>

#remarque([Les données de l'essai B semblent absurde, ce qui conduit à penser à une erreur de manipulation ou de mesure. On ne les prendra pas en compte par la suite.])

#figure(
  image("assets/ponts_arc.png"),
  caption: [Photo du ponts en arc vu de côté]
)

== Modélisation mécanique

=== Liaisons

Nous avons choisi de modéliser les appuis du tablier par des appuis fixes car nous avons remarqué que le tablier pouvait s'arquer. Le couple imposé par les appuis est ainsi presque nul, ce qui est en adéquation avec le choix d'une rotule. De plus, les appuis empêchent le pont de se translater selon $arrow(e)_x$ et selon $arrow(e)_y$, ce qui nous a conduit à modéliser les appuis par des appuis fixes.

De la même façon, la liaison entre le tablier et l'arche a été modélisée par des rotules : l'arche peut tourner selon $arrow(e)_z$ mais ne peut pas se déplacer selon $arrow(e)_x$ ou $arrow(e)_y$ car elle est collée au tablier.

Les liaisons entre le pilier et le tablier d'une part, et entre le pilier et l'arche d'autre part, ont toutes deux été modélisées par des encastrements, car le pilier est suffisamment épais, rigide et bien collé pour empêcher déplacements et rotations.

=== Hypothèses

- *Petites déformations* : les déformations et les déplacements sont très petits devant la longueur du pont. Cela a été vérifié expérimentalement.
- *Navier--Bernoulli* : la portée des éléments de structure est très grande devant leur hauteur. C'est vérifié, l'épaisseur du papier étant très petite devant la longueur du pont.
- Le poids du papier est négligé devant celui des rondelles métalliques.
- Le papier reste dans son domaine élastique.

=== Calculs théoriques

Pour calculer la raideur totale du pont, on le décompose en trois sous-ensembles : l'arche, le pilier et le tablier.

#figure(
  image("assets/model_arc_forces.svg"),
  caption: [Modélisation des forces intérieures.]
) <fig:model-arc-forces>

*Tablier.* Sa raideur est celle trouvée dans la première partie : le tablier est posé sur deux appuis fixes et soumis en son centre à une force verticale $N$, exercée par le pilier. Ainsi
$ N = (48 E I) / L^3 delta_"tab", $
où $delta_"tab"$ est la flèche du tablier. On rappelle que $k_"tab" = (48E I)/(L^3) = (4 E l h^3)/(L^3)$

*Arche.* Sa raideur nous est donnée par l'énoncé. En modélisant les appuis de l'arche par des appuis fixes :
$ F - N = (2048 E I) / L^3 delta_"arche". $

*Pilier.* Montrons que sa raideur est très grande devant celle du tablier. Soit $epsilon$ la déformation longitudinale du pilier :
$ epsilon = N / (E A) = xi'_t. $
En intégrant le long du pilier, $xi_t = N h_p \/ (E A)$, d'où la raideur du pilier
$ k_"pilier" = (E A) / h_p. $
Le rapport des raideurs vaut, avec $A = pi r_"pilier"^2$ et $I = l h^3 \/ 12$,
$ k_"pilier" / k_"tab" = (E A) / h_p dot L^3 / (48 E I) = (A L^3) / (48 h_p I) = pi / 4 (r_"pilier" / h)^2 dot L / h_p dot L / l dot L / h. $
Or $r_"pilier" >> h$ et $L >> h$, tandis que $L \/ h_p$ et $L \/ l$ sont de l'ordre de l'unité. Ainsi
$ k_"pilier" / k_"tab" >> 1. $

On néglige donc la déformation du pilier devant celle du tablier : tout le déplacement transmis par l'arche au pilier est retransmis au tablier, et $delta_"tab" = delta_"arche" = delta$. On a alors
$ F = ((2048 E I) / L^3 + (48 E I) / L^3) delta. $

La raideur totale du pont est donc
$ k_"tot" = (2096 E I) / L^3 = (524 l h^3) / (3 L^3) E = 2096/48 k$
Avec la raideur $k$ mesurée en première partie, l'application numérique donne $k_"tot" = #arrondi(modele(5)) "N/m"$.

== Critique du modèle

#figure(
  image("assets/exp5_cloche.png", width: 75%),
  caption: [Déformation en fonction de la force pour le pont en arc.],
) <fig:exp5>

L'essai A donne $k_"tot" = #pm(k(5), unite: "N/m")$, soit un écart de $#calc.round((k(5).valeur / modele(5).valeur - 1) * 100) %$ avec le résultat théorique.
Il semble que le modèle semble moins bien décrire le phénomène.
