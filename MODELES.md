# Modèles d'authentification — comparaison et choix

Annexe méthodologique : pourquoi le score d'authenticité de l'étape 2 du
pipeline de vérification (`app/routers/verification.py`) est produit par
EfficientNet-B0 (`luxury_authenticator_v1/`) plutôt que par le Random Forest
historique (`app/ml/`), et dans quelles limites.

## Méthode de comparaison

Les deux modèles ont été évalués sur le **même jeu de test réel**, celui
utilisé pour entraîner et évaluer l'authenticator : 235 images, 106
contrefaçons et 129 authentiques. Le Random Forest n'a pas été ré-entraîné
pour cette comparaison — il s'agit du modèle tel qu'il tourne en production
aujourd'hui, confronté à des images réelles qu'il n'a jamais vues à
l'entraînement (son propre jeu d'entraînement était synthétique, voir plus
bas).

## Résultats

### Random Forest actuel (`app/ml/`)

| Métrique | Valeur |
|---|---|
| AUC ROC | 0,534 |
| Accuracy (seuil 0,5) | 57,9 % |
| Score moyen — contrefaçons | 0,703 |
| Score moyen — authentiques | 0,731 |

Au seuil 0,5, **99 contrefaçons sur 106 sont classées comme authentiques**.
Un AUC de 0,534 est à peine au-dessus du hasard (0,5) : le modèle ne
sépare pas les deux classes sur des images réelles. Relever le seuil ne
corrige pas le problème : à 0,85, l'accuracy tombe à 44,7 % et le rappel
(capacité à détecter les vraies contrefaçons) s'effondre à 9,3 %.

### EfficientNet-B0 (`luxury_authenticator_v1/`)

| Métrique | Valeur |
|---|---|
| AUC ROC | 0,899 |
| Accuracy | 82,6 % |
| Précision | 83,8 % |
| Rappel | 84,5 % |
| F1 | 84,2 % |

Matrice de confusion (235 images) :

| | Prédit contrefaçon | Prédit authentique |
|---|---|---|
| **Réel contrefaçon** (106) | 85 détectées | 21 classées authentiques à tort |
| **Réel authentique** (129) | 20 rejetés à tort | 109 validés |

## Composition du jeu de test

| | Contrefaçons | Authentiques |
|---|---|---|
| Nike | 33 | 38 |
| Gucci | 24 | 31 |
| Adidas | 19 | 28 |
| Puma | 15 | 29 |
| Louis Vuitton | 15 | 3 |

Environ 60 % des images sont des captures d'écran (et non des photos
originales), le reste des photos produit.

## Limites — à lire avant toute présentation des résultats

- **Déséquilibre Louis Vuitton** : 15 contrefaçons contre seulement 3
  authentiques pour cette marque — la classe authentique y est trop peu
  représentée pour tirer une conclusion par marque.
- **Confusion possible provenance / authenticité** : les dimensions
  d'image diffèrent structurellement entre les deux classes (contrefaçons
  souvent en 1280×720, authentiques dans des formats plus grands). Le
  modèle a pu apprendre en partie à reconnaître la *provenance* de l'image
  (capture d'écran d'une fiche produit contrefaite vs. photo originale)
  plutôt que l'authenticité du produit lui-même. Les métriques ci-dessus
  restent donc une mesure de performance sur ce jeu précis, pas une preuve
  que le modèle généralise à des photos prises dans des conditions
  différentes.
- **Jeu d'entraînement du Random Forest synthétique** : ses métriques
  historiques (accuracy 85,8 % citées dans l'ancien écran « Modèle IA »)
  ont été mesurées sur des données synthétiques, pas sur des photos
  réelles — elles ne sont pas comparables aux chiffres ci-dessus et ne
  doivent plus être présentées comme représentatives.
- **Couverture par catégorie** : le jeu d'entraînement de l'authenticator
  est très majoritairement composé de prêt-à-porter et de sneakers, avec
  une part limitée de maroquinerie de luxe et aucune montre ni bijou.
  L'extension à ces catégories demandera un jeu de données spécialisé.
- **Aide à la décision, pas une certification** : le résultat du modèle
  reste un signal parmi sept dans le score pondéré final (poids 0,22 pour
  la composante authenticité), et la zone d'incertitude (score global
  entre 0,40 et 0,84) est systématiquement arbitrée par un expert humain.

## Décision retenue

EfficientNet-B0 remplace le Random Forest comme source du score
d'authenticité de l'étape 2. Le Random Forest n'est pas supprimé : il
reste le repli automatique si EfficientNet ne peut pas être chargé
(fichier de poids absent, `torch`/`torchvision` indisponibles, mémoire
insuffisante...), et sert de point de comparaison historique documenté
ci-dessus. Aucun des deux modèles ne remplace le jugement de l'expert
humain sur la zone d'incertitude.

## Ce qui est versionné dans `luxury_authenticator_v1/`

Le dossier reproduit le livrable tel que reçu : scripts de téléchargement
et de préparation des données (`data/`), entraînement et évaluation
(`training/`), inférence (`inference/predict.py` — seul module réellement
utilisé par l'application principale, via `app/ml/authenticator.py`), et
une petite API/interface web de démonstration autonome (`app/main.py`,
`frontend/`) qui n'est **pas déployée** : conformément à la décision
d'architecture, LUXLOC reste une seule application, et EfficientNet est
consommé en bibliothèque, pas comme un second service.
