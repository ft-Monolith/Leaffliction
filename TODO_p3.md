# TODO Part 3 — Transformation.py (à finir)

État au 2026-07-07 : le module marche, flake8 OK, les 2 modes (image / batch
`-src -dst -mask`) fonctionnent, `-h` OK. Audit visuel des transfos fait via
planches de contrôle (1 image par classe, 8 classes) — voir section 1/2.

## 1. Auditer chaque transfo vs la figure du sujet

- [x] **Gaussian blur (IV.2)** — CORRIGÉ. Utilisait un seuil *saturation* qui
      échouait sur les feuilles peu saturées (Apple_healthy quasi vide). Passé
      au même seuil LAB 'a' que le masque → silhouette floutée propre et
      cohérente sur les 8 classes. (Les trous sur Grape_Black_rot/Esca = les
      taches de maladie sous le seuil, avant fill_holes : effet correct.)
- [x] **Mask (IV.3)** — DÉCISION : on garde propre (feuille isolée pleine en
      couleur). Correspond à la figure IV.3 du sujet (feuille isolée). Validé
      sur les 8 classes, segmentation solide partout.
- [x] **Roi objects (IV.4)** — SIMPLIFIÉ (audit 2026-07-07) : vert =
      feuille entière (masque gardé par pcv.roi.filter) + cadre bleu.
      Le 2e seuil saturation "tissu sain" a été retiré (dur à justifier
      en défense, purement esthétique). Matche la figure IV.4.
- [x] **Analyze object (IV.5)** — contour rouge + lignes de forme magenta,
      suit bien la feuille sur les 8 classes. Acceptable.
- [x] **Pseudolandmarks (IV.6)** — points bien placés (contour + nervure
      centrale), couleurs distinctes. OK sur les 8 classes.
- [x] **Color histogram (IV.7)** — 9 canaux, proportion %, courbes séparées +
      légende. Matche la figure IV.7. OK.

## 2. Robustesse sur TOUTES les classes (important)

- [x] Masque (LAB 'a' + otsu) validé visuellement sur les **8 classes**
      (Apple_Black_rot/healthy/rust/scab, Grape_Black_rot/Esca/healthy/spot).
      Segmentation solide partout, y compris les raisins sombres. Pas de cas
      qui foire.
- [x] Planche de contrôle générée (1 image/classe × transfos clés) et revue.

## 3. Mode affichage (à tester sur machine avec écran)

- [ ] Lancer `cd srcs && ./Transformation.py "<image>"` avec écran : vérifier
      les 2 fenêtres (grille 2x3 + histogramme), lisibilité, titres.

## 4. Divers

- [x] Shebang `#!/usr/bin/env python3` + docstring module ajoutés (audit
      2026-07-07) — `./Transformation.py -h` refonctionne depuis srcs/.

- [x] Comportement de `-mask` : sauve en plus le masque binaire
      `_Leaf_mask.JPG`. Sémantique conservée. OK.
- [x] Batch : retesté sur dossier hétérogène (2 images valides + .txt + JPEG
      cassé) → 2/3 traités, le cassé sauté proprement, pas de crash.
- [x] `flake8 setup.py srcs/*.py` clean. (Restent des erreurs dans
      `srcs/classification/*.py` = part 4 du mate, pas à moi.)
- [x] `yes/` supprimé.

## Rappels archi (pour ne pas se retromper)

- Vrai code dans `srcs/` ; lancé depuis `srcs/` (`./Transformation.py ...`).
- Sous-commande aussi dispo : `python setup.py transform ...` (depuis racine).
- Dossier = `-src`/`-dst` (PAS un argument nu). Argument nu = 1 image à afficher.
