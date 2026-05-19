"""
LUXLOC — Extraction de features pour le modèle de détection de contrefaçon
═════════════════════════════════════════════════════════════════════════

Transforme une image d'article en un vecteur de features numériques
exploitable par un modèle de machine learning.

Les features capturent les signaux réels qui distinguent un article
authentique d'une contrefaçon :
  - netteté et qualité photographique (un faux est souvent moins soigné)
  - contraste et luminosité
  - richesse et régularité des contours (coutures, finitions)
  - texture du matériau (grain du cuir vs surface plastifiée)

Toutes les features sont calculées à partir des pixels réels de l'image,
en réutilisant le module cv_analysis. Ce sont les mêmes descripteurs que
ceux d'un pipeline de vision industriel.
"""
import numpy as np
from app.services import cv_analysis as cv

# Ordre des features — doit rester stable (le modèle est entraîné dessus)
FEATURE_NAMES = [
    "sharpness",          # variance du Laplacien (netteté)
    "brightness",         # luminosité moyenne
    "contrast",           # écart-type des niveaux de gris
    "edge_density",       # proportion de pixels-contours (Sobel)
    "edge_strength",      # intensité moyenne des contours
    "glcm_contrast",      # contraste de texture (GLCM)
    "glcm_homogeneity",   # homogénéité de texture (GLCM)
    "glcm_energy",        # énergie de texture (GLCM)
    "texture_entropy",    # entropie des niveaux de gris (désordre du grain)
]


def extract_features(img):
    """
    Extrait le vecteur de features d'une image PIL.
    Renvoie un np.array de longueur len(FEATURE_NAMES).
    """
    gray = cv.to_gray_array(img)

    # 1-3 — netteté, luminosité, contraste
    sharpness = cv.laplacian_variance(gray)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # 4-5 — contours (Sobel)
    edges = cv.sobel_edges(gray)
    thr = np.mean(edges) + np.std(edges)
    edge_density = float(np.mean(edges > thr))
    edge_strength = float(np.mean(edges))

    # 6-8 — texture (GLCM)
    glcm = cv.glcm_features(gray)

    # 9 — entropie des niveaux de gris (mesure du désordre du grain)
    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) else 0.0

    return np.array([
        sharpness,
        brightness,
        contrast,
        edge_density,
        edge_strength,
        glcm["contrast"],
        glcm["homogeneity"],
        glcm["energy"],
        entropy,
    ], dtype=np.float64)


def features_to_dict(vec):
    """Associe un vecteur de features à ses noms (pour l'affichage / l'API)."""
    return {name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, vec)}
