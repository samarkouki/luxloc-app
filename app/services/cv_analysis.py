"""
LUXLOC — Module d'analyse d'image (Computer Vision)
═══════════════════════════════════════════════════

Analyse RÉELLE des images uploadées pour le pipeline de vérification IA.
Utilise Pillow + numpy uniquement (pas de dépendance lourde).

Chaque fonction renvoie un score normalisé entre 0 et 1 calculé à partir
des propriétés réelles de l'image, pas d'une valeur aléatoire.

En production, les étapes CNN seront remplacées par des modèles entraînés
(EfficientNet-B4, Siamese Network) servis via Vertex AI. Les étapes
analytiques ci-dessous (flou, GLCM, contours, pHash) restent valables
telles quelles — ce sont de vraies techniques de traitement d'image.
"""
import base64
import io
import math
from PIL import Image
import numpy as np


# ════════════════════════════════════════
# DÉCODAGE
# ════════════════════════════════════════

def decode_image(data_uri_or_b64):
    """
    Décode une image depuis une data URI (data:image/jpeg;base64,...)
    ou une chaîne base64 brute. Renvoie un objet PIL.Image en RGB.
    """
    if data_uri_or_b64 is None:
        return None
    s = data_uri_or_b64
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        raw = base64.b64decode(s)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception:
        return None


def to_gray_array(img):
    """Convertit une image PIL en tableau numpy niveaux de gris (float 0-255)."""
    return np.asarray(img.convert("L"), dtype=np.float64)


# ════════════════════════════════════════
# ÉTAPE 1 — CONTRÔLE QUALITÉ (réel)
# ════════════════════════════════════════

def laplacian_variance(gray):
    """
    Variance du Laplacien — mesure de netteté.
    Une image floue a une faible variance (peu de transitions brusques).
    Technique standard de détection de flou (Pech-Pacheco et al., 2000).
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = gray.shape
    lap = np.zeros_like(gray)
    # Convolution 3x3
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            k = kernel[dy + 1, dx + 1]
            if k == 0:
                continue
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lap += k * shifted
    inner = lap[1:h-1, 1:w-1]
    return float(np.var(inner))


def quality_check(img):
    """
    Contrôle qualité réel : résolution, netteté, luminosité, contraste.
    Renvoie un dict avec les mesures et un verdict 'passed'.
    """
    w, h = img.size
    gray = to_gray_array(img)

    lap_var = laplacian_variance(gray)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Seuils calibrés pour des photos de smartphone
    resolution_ok = (w >= 200 and h >= 200)
    sharpness_ok = lap_var > 80.0          # < 80 = image floue
    brightness_ok = 35.0 < brightness < 230.0  # ni trop sombre ni cramée
    contrast_ok = contrast > 18.0          # image plate = peu de détails

    passed = resolution_ok and sharpness_ok and brightness_ok and contrast_ok

    return {
        "width": w,
        "height": h,
        "laplacian_variance": round(lap_var, 2),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "resolution_ok": resolution_ok,
        "sharpness_ok": sharpness_ok,
        "brightness_ok": brightness_ok,
        "contrast_ok": contrast_ok,
        "passed": passed,
    }


# ════════════════════════════════════════
# CONTOURS (Sobel) — utilisé étapes 2, 3a, 3b
# ════════════════════════════════════════

def sobel_edges(gray):
    """Magnitude du gradient de Sobel — carte des contours."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    def conv(g, k):
        out = np.zeros_like(g)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                c = k[dy + 1, dx + 1]
                if c == 0:
                    continue
                out += c * np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        return out

    gx = conv(gray, kx)
    gy = conv(gray, ky)
    return np.sqrt(gx ** 2 + gy ** 2)


def edge_density(gray):
    """Proportion de pixels appartenant à un contour franc (0-1)."""
    edges = sobel_edges(gray)
    threshold = np.mean(edges) + np.std(edges)
    return float(np.mean(edges > threshold))


# ════════════════════════════════════════
# ÉTAPE 2 — SCORE GLOBAL (réel, basé features)
# ════════════════════════════════════════

def global_quality_score(img):
    """
    Score global basé sur des features réelles de l'image.

    NOTE : en production, cette fonction est remplacée par EfficientNet-B4
    entraîné sur un dataset authentique/contrefaçon. Ici on calcule un
    proxy déterministe à partir de la netteté, du contraste et de la
    richesse en contours — des indicateurs corrélés à la qualité d'une
    photo d'article authentique bien présenté.
    """
    gray = to_gray_array(img)
    lap_var = laplacian_variance(gray)
    contrast = float(np.std(gray))
    e_density = edge_density(gray)

    # Normalisation de chaque feature vers 0-1
    f_sharp = min(lap_var / 600.0, 1.0)        # netteté
    f_contrast = min(contrast / 70.0, 1.0)     # contraste
    f_edges = min(e_density / 0.18, 1.0)       # richesse en détails

    score = 0.45 * f_sharp + 0.30 * f_contrast + 0.25 * f_edges
    return float(np.clip(score, 0.0, 1.0))


# ════════════════════════════════════════
# ÉTAPE 3a — ANALYSE LOGO (réel)
# ════════════════════════════════════════

def logo_score(img):
    """
    Analyse du logo : netteté et contraste local dans la zone centrale.
    Un vrai logo de luxe est net, bien défini, avec des contours francs.
    En production : Siamese Network comparant au logo officiel de la marque.
    """
    gray = to_gray_array(img)
    h, w = gray.shape
    # Zone centrale (le logo est généralement au centre du crop)
    cy, cx = h // 2, w // 2
    ch, cw = h // 3, w // 3
    center = gray[max(0, cy-ch):cy+ch, max(0, cx-cw):cx+cw]
    if center.size == 0:
        center = gray

    sharp = min(laplacian_variance(center) / 500.0, 1.0)
    contrast = min(float(np.std(center)) / 65.0, 1.0)
    e_density = min(edge_density(center) / 0.20, 1.0)

    score = 0.40 * sharp + 0.30 * contrast + 0.30 * e_density
    return float(np.clip(score, 0.0, 1.0))


# ════════════════════════════════════════
# ÉTAPE 3b — ANALYSE COUTURES (réel)
# ════════════════════════════════════════

def stitching_score(img):
    """
    Analyse de la régularité des coutures.

    Principe : sur une vraie couture de luxe, les points sont régulièrement
    espacés. On projette les contours sur l'axe horizontal et on mesure la
    périodicité du signal : un signal régulier (faible coefficient de
    variation des intervalles entre pics) = couture régulière.
    """
    gray = to_gray_array(img)
    edges = sobel_edges(gray)

    # Profil de contours : somme des contours par colonne
    profile = np.sum(edges, axis=0)
    if profile.size < 10:
        return 0.5
    # Normalisation
    profile = (profile - profile.min()) / (np.ptp(profile) + 1e-6)

    # Détection des pics (points de couture)
    mean_p = np.mean(profile)
    peaks = []
    for i in range(1, len(profile) - 1):
        if profile[i] > mean_p and profile[i] >= profile[i-1] and profile[i] > profile[i+1]:
            peaks.append(i)

    if len(peaks) < 3:
        # Peu de pics détectés : on se rabat sur la régularité globale de la texture
        return float(np.clip(0.55 + 0.3 * (1 - np.std(profile)), 0.0, 1.0))

    # Intervalles entre pics
    intervals = np.diff(peaks)
    cv = float(np.std(intervals) / (np.mean(intervals) + 1e-6))  # coefficient de variation

    # Faible CV = espacement régulier = couture authentique
    regularity = max(0.0, 1.0 - cv)
    return float(np.clip(0.30 + 0.70 * regularity, 0.0, 1.0))


# ════════════════════════════════════════
# ÉTAPE 3c — ANALYSE TEXTURE (GLCM réel)
# ════════════════════════════════════════

def glcm_features(gray, levels=16, distance=1):
    """
    Gray-Level Co-occurrence Matrix (Haralick, 1973).
    Calcule la matrice de co-occurrence puis extrait 3 descripteurs réels :
    contraste, homogénéité, énergie.
    """
    # Quantification sur 'levels' niveaux
    q = (gray / 256.0 * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1)

    glcm = np.zeros((levels, levels), dtype=np.float64)
    # Co-occurrences horizontales (décalage = distance)
    a = q[:, :-distance].ravel()
    b = q[:, distance:].ravel()
    for i, j in zip(a, b):
        glcm[i, j] += 1
    # Symétrisation + normalisation
    glcm = glcm + glcm.T
    total = glcm.sum()
    if total == 0:
        return {"contrast": 0.0, "homogeneity": 0.0, "energy": 0.0}
    glcm /= total

    idx = np.arange(levels)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    contrast = float(np.sum(glcm * (I - J) ** 2))
    homogeneity = float(np.sum(glcm / (1.0 + (I - J) ** 2)))
    energy = float(np.sum(glcm ** 2))
    return {"contrast": contrast, "homogeneity": homogeneity, "energy": energy}


def texture_score(img):
    """
    Score de texture basé sur les features GLCM réelles.

    Le cuir / tissu d'un article de luxe authentique présente un grain
    structuré : homogénéité modérée (texture présente mais pas chaotique)
    et énergie ni trop faible (texture absente) ni trop forte (surface
    plastifiée uniforme typique des contrefaçons).
    """
    gray = to_gray_array(img)
    feat = glcm_features(gray)

    # Une texture authentique a une homogénéité dans une plage saine
    homog = feat["homogeneity"]
    energy = feat["energy"]

    # Score = homogénéité dans [0.2, 0.7] est idéale ; énergie pénalisée si extrême
    f_homog = 1.0 - abs(homog - 0.45) / 0.45
    f_energy = 1.0 - min(abs(energy - 0.15) / 0.30, 1.0)

    score = 0.6 * np.clip(f_homog, 0, 1) + 0.4 * np.clip(f_energy, 0, 1)
    return float(np.clip(score, 0.0, 1.0)), feat


# ════════════════════════════════════════
# ÉTAPE 3d — DÉTECTION ZONE ÉTIQUETTE (réel)
# ════════════════════════════════════════

def label_score(img):
    """
    Détection de la présence d'une zone texte (étiquette / numéro de série).

    Sans moteur OCR installé, on détecte les zones à forte densité de
    contours fins et alignés — signature d'un texte imprimé. Une vraie
    étiquette présente un bloc de texte net et contrasté.
    En production : Google Vision API extrait le texte + regex format marque.
    """
    gray = to_gray_array(img)
    edges = sobel_edges(gray)
    h, w = edges.shape

    # On découpe l'image en bandes horizontales et on cherche
    # la bande la plus dense en contours (le bloc de texte)
    bands = 8
    band_h = max(1, h // bands)
    densities = []
    for b in range(bands):
        strip = edges[b*band_h:(b+1)*band_h, :]
        if strip.size:
            thr = np.mean(edges) + 0.5 * np.std(edges)
            densities.append(float(np.mean(strip > thr)))
    if not densities:
        return 0.5

    max_density = max(densities)
    contrast = float(np.std(gray))

    f_text = min(max_density / 0.25, 1.0)        # présence d'un bloc texte net
    f_contrast = min(contrast / 60.0, 1.0)       # étiquette lisible
    score = 0.65 * f_text + 0.35 * f_contrast
    return float(np.clip(score, 0.0, 1.0))


# ════════════════════════════════════════
# HASH PERCEPTUEL (anti-doublon / base de faux)
# ════════════════════════════════════════

def perceptual_hash(img):
    """
    pHash 64 bits (algorithme aHash simplifié).
    Permet de détecter qu'une photo a déjà été soumise (réutilisation
    d'images de contrefaçons connues).
    """
    small = img.convert("L").resize((8, 8), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float64)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return format(h, "016x")


def hamming_distance(hash1, hash2):
    """Distance de Hamming entre deux pHash (nombre de bits différents)."""
    x = int(hash1, 16) ^ int(hash2, 16)
    return bin(x).count("1")
