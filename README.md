# Leaffliction

Projet de vision par ordinateur : classification de maladies de plantes à partir
de photographies de feuilles.

Le projet couvre l'ensemble de la chaîne, de l'analyse du jeu de données brut
jusqu'au classifieur entraîné : graphiques de répartition des classes,
augmentation de données pour équilibrer le jeu, transformations d'images basées
sur PlantCV (segmentation de la feuille, ROI, analyse de forme,
pseudolandmarks, histogramme de couleurs), et un réseau de neurones
convolutionnel entraîné avec PyTorch qui prédit la maladie d'une feuille donnée.

Le sujet complet est disponible dans [en.subject.pdf](en.subject.pdf).

## Prérequis

- Python 3.10 ou plus récent
- Le jeu de données de feuilles (`leaves.zip` fourni par le sujet), décompressé
  dans un dossier dont les sous-dossiers sont les classes :

```
images/
├── apple_healthy/
├── apple_scab/
├── apple_black_rot/
└── apple_rust/
```

Le jeu de données, les images générées, le modèle et les archives ne sont pas
versionnés (voir [.gitignore](.gitignore)).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

Toutes les commandes sont exposées par [setup.py](setup.py) :

```bash
python setup.py <commande> [arguments]
```

`train.py` et `predict.py` sont de simples raccourcis vers les sous-commandes
`train` et `predict` : `python train.py ./images` et
`python setup.py train ./images` sont équivalents.

### Analyse du jeu de données

Affiche un diagramme circulaire et un diagramme en barres du nombre d'images
par classe.

```bash
python setup.py distribution ./images
```

### Augmentation de données

Sur une image, génère et affiche six augmentations (flip, rotation, skew,
shear, crop, distorsion) et les enregistre à côté du fichier d'origine, sous le
nom original suivi du type d'augmentation.

```bash
python setup.py augment "./images/apple_healthy/image (1).JPG"
```

Sur un dossier, équilibre chaque classe en générant des images augmentées
jusqu'à ce que toutes atteignent la taille de la plus grande.

```bash
python setup.py augment ./images
```

### Transformation d'images

Sur une image, affiche l'originale, un flou gaussien du masque de la feuille,
la feuille masquée, les objets de la ROI, l'analyse de forme, les
pseudolandmarks et l'histogramme de couleurs sur neuf canaux (RGB, HSV, LAB).

```bash
python setup.py transform "./images/apple_healthy/image (1).JPG"
```

Avec un dossier source et un dossier de destination, enregistre toutes les
transformations de toutes les images au lieu de les afficher. L'option `-mask`
enregistre également le masque de la feuille isolée.

```bash
python setup.py transform -src ./images/apple_healthy -dst ./transformed -mask
```

### Entraînement

```bash
python setup.py train ./images
```

La commande :

1. copie le jeu de données dans `augmented_directory/` (réutilisé s'il existe
   déjà), l'équilibre à l'aide des méthodes d'augmentation, puis remplace
   chaque image par sa version masquée ;
2. sépare les données en un ensemble d'entraînement et un ensemble de
   validation (20 %, au minimum 100 images, graine fixe) ;
3. entraîne le CNN et enregistre le meilleur modèle, selon la précision de
   validation, dans `models/model.pth` ;
4. produit `directory.zip`, contenant le modèle et le jeu de données augmenté,
   et écrit son empreinte SHA-1 dans `signature.txt`.

### Évaluation

Extrait `directory.zip` dans `unzipped/` et affiche la précision du modèle sur
l'ensemble de validation.

```bash
python setup.py evaluate directory.zip
```

### Prédiction

Affiche l'image originale et l'image transformée, et affiche la classe prédite.

```bash
python setup.py predict "./images/apple_healthy/image (1).JPG"
```

Le modèle utilisé par défaut est `unzipped/model.pth`, celui extrait par la
commande `evaluate` ; un autre chemin peut être passé en second argument. Les
noms de classes sont lus dans les sous-dossiers de `images/`, qui doit donc
être présent et identique au jeu de données utilisé pour l'entraînement.

### Jeu de données masqué

Écrit une copie d'un jeu de données dans laquelle chaque image est remplacée
par sa feuille masquée. Cette étape est déjà effectuée par `train` ; elle n'est
exposée que pour inspection.

```bash
python setup.py mask ./images ./masked
```

### Archive

Reconstruit `directory.zip` et `signature.txt` à partir d'un modèle et d'un jeu
de données existants.

```bash
python setup.py zip ./augmented_directory models/model.pth
```

## Modèle

`LeafCNN` ([srcs/classification/cnn.py](srcs/classification/cnn.py)) est un
petit réseau convolutionnel entraîné sur des images de feuilles masquées en
64x64 :

- trois couches de convolution (3 vers 64, 64 vers 128, 128 vers 256 canaux),
  chacune suivie d'un ReLU, les deux premières d'un max pooling 2x2 ;
- un global average pooling réduisant chaque carte de caractéristiques à une
  seule valeur ;
- une couche entièrement connectée projetant les 256 caractéristiques sur les
  classes.

L'entraînement utilise une entropie croisée et l'optimiseur Adam (taux
d'apprentissage 0.001), sur 50 époques au maximum avec un arrêt anticipé après
5 époques sans amélioration de la précision de validation. CUDA et Apple MPS
sont utilisés lorsqu'ils sont disponibles, sinon le CPU.

## Structure du projet

```
setup.py                        point d'entrée en ligne de commande
train.py, predict.py            raccourcis vers les commandes train et predict
srcs/
├── Distribution.py             graphiques de répartition des classes
├── Augmentation.py             les six augmentations, équilibrage du jeu
├── Transformation.py           transformations PlantCV et histogramme
├── utils.py                    fonctions utilitaires partagées
└── classification/
    ├── cnn.py                  modèle, choix du device, chargement
    ├── dataset.py              dataset, prétraitement, découpage train/val
    ├── mask_dataset.py         copie masquée d'un jeu de données
    ├── train.py                boucle d'entraînement
    ├── predict.py              prédiction sur une image
    ├── evaluate.py             précision d'un modèle empaqueté dans un zip
    └── zip.py                  archive et signature SHA-1
```

`Distribution.py`, `Augmentation.py` et `Transformation.py` peuvent également
être lancés comme des scripts autonomes :

```bash
python srcs/Distribution.py ./images
```

## Norme

Le projet suit la norme Python de 42, PEP 8 vérifiée avec flake8 :

```bash
flake8 .
```
