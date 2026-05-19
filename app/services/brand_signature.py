"""
LUXLOC — Détection de signature de marque
═════════════════════════════════════════

Vérifie qu'une photo d'article présente la signature visuelle attendue
de sa marque. Premier détecteur implémenté : Tommy Hilfiger.

Tommy Hilfiger possède un logo « flag » iconique combinant trois couleurs
précises : un rouge franc, un bleu marine profond et du blanc. Les
contrefaçons se trahissent souvent par des couleurs incorrectes — un
rouge trop orangé, un bleu délavé. Ce module analyse la palette réelle
de l'image et mesure à quel point elle correspond à la signature
officielle de la marque.

Technique : analyse de la palette de couleurs (numpy). Réelle et
déterministe. Ce n'est pas un classifieur entraîné mais une vérification
de cohérence chromatique — un signal complémentaire au modèle ML.
"""
import numpy as np
from PIL import Image


# ════════════════════════════════════════
# RÉFÉRENCES COULEUR — TOMMY HILFIGER
# ════════════════════════════════════════
# Couleurs officielles du logo « flag » Tommy Hilfiger
TOMMY_RED = np.array([200, 16, 46], dtype=np.float64)    # rouge Tommy (~#C8102E)
TOMMY_NAVY = np.array([0, 31, 92], dtype=np.float64)     # marine Tommy (~#001F5C)
TOMMY_WHITE = np.array([245, 245, 245], dtype=np.float64)

# Distance maximale tolérée dans l'espace RGB pour considérer une couleur
# « correcte » (calibrée empiriquement)
MAX_RGB_DIST = 110.0


def _downscale(img, max_side=300):
    """Réduit l'image pour accélérer l'analyse de palette."""
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def _mask_reddish(arr):
    """Pixels nettement rouges (R dominant, vert bas — exclut les tons dorés)."""
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    return (R > 90) & (R > G + 35) & (R > B + 25) & (G < 130)


def _mask_navyish(arr):
    """Pixels bleu foncé (B dominant et image sombre)."""
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    return (B > 45) & (B > R + 20) & (B > G + 10) & (R < 120) & (G < 130)


def _mask_white(arr):
    """Pixels blancs / quasi blancs."""
    return arr.min(axis=-1) > 195


def _mask_gold(arr):
    """Pixels dorés métalliques (logo embossé or)."""
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    return (R > 165) & (G > 120) & (G < 200) & (B < 130) & \
           (R > G + 12) & (G > B + 30) & (R - B > 80)


def _closeness(mean_color, target):
    """Renvoie une proximité 0-1 entre une couleur moyenne et une cible."""
    dist = float(np.linalg.norm(mean_color - target))
    return max(0.0, 1.0 - dist / MAX_RGB_DIST)


def tommy_hilfiger_signature(img):
    """
    Analyse la signature visuelle Tommy Hilfiger d'une image.

    Tommy Hilfiger décline son logo en plusieurs variantes officielles :
      - le logo « flag » tricolore (rouge / blanc / marine),
      - une version monochrome dorée (embossée, quincaillerie),
      - une version monochrome argentée / chromée.

    Portée réelle du détecteur :
      - le flag tricolore est vérifié de façon fiable, teintes comprises ;
      - une dorure métallique est détectable comme indice positif modéré ;
      - une version argentée n'est PAS confirmable par analyse couleur
        seule (un gris métallique est indistinguable d'un gris neutre).
    Quand la signature n'est pas identifiable, le détecteur reste NEUTRE
    et ne pénalise pas l'article — l'inspection est laissée à l'expert.

    Renvoie : score 0-1, verdict, détails, et un drapeau 'conclusif'.
    """
    arr = _downscale(img)
    total = arr.shape[0] * arr.shape[1]

    m_red = _mask_reddish(arr)
    m_navy = _mask_navyish(arr)
    m_white = _mask_white(arr)
    m_gold = _mask_gold(arr)

    frac_red = float(m_red.sum()) / total
    frac_navy = float(m_navy.sum()) / total
    frac_white = float(m_white.sum()) / total
    frac_gold = float(m_gold.sum()) / total

    rouge_present = frac_red > 0.004
    marine_present = frac_navy > 0.004
    blanc_present = frac_white > 0.01
    dore_present = frac_gold > 0.02

    details = {
        "rouge_present": rouge_present,
        "marine_present": marine_present,
        "blanc_present": blanc_present,
        "dore_present": dore_present,
        "part_rouge": round(frac_red, 4),
        "part_marine": round(frac_navy, 4),
        "part_dore": round(frac_gold, 4),
    }

    # ── Variante 1 : logo flag tricolore (vérification fiable) ───────
    if rouge_present or marine_present:
        presents = sum([rouge_present, marine_present, blanc_present])
        presence = presents / 3.0
        justesses = []
        if rouge_present:
            red_ok = _closeness(arr[m_red].mean(axis=0), TOMMY_RED)
            details["justesse_rouge"] = round(red_ok, 3)
            justesses.append(red_ok)
        if marine_present:
            navy_ok = _closeness(arr[m_navy].mean(axis=0), TOMMY_NAVY)
            details["justesse_marine"] = round(navy_ok, 3)
            justesses.append(navy_ok)
        justesse = float(np.mean(justesses)) if justesses else 0.5
        score = 0.40 * presence + 0.60 * justesse
        details["variante_detectee"] = "flag tricolore"
        if score >= 0.75:
            verdict = "Signature flag tricolore conforme à Tommy Hilfiger"
        elif score >= 0.50:
            verdict = "Flag tricolore présent — teintes à vérifier"
        else:
            verdict = "Teintes du flag incohérentes — contrefaçon probable"
        return {"score": round(float(np.clip(score, 0, 1)), 3),
                "verdict": verdict, "conclusif": True, "details": details}

    # ── Variante 2 : logo monochrome doré (indice positif modéré) ────
    if dore_present:
        score = min(0.74 + frac_gold * 3.0, 0.90)
        details["variante_detectee"] = "monochrome doré"
        return {"score": round(score, 3),
                "verdict": "Logo doré métallique détecté — cohérent Tommy Hilfiger",
                "conclusif": True, "details": details}

    # ── Aucune variante identifiable : neutre, non pénalisant ────────
    details["variante_detectee"] = None
    return {
        "score": 0.6,
        "verdict": "Logo non identifiable par couleur (variante monochrome possible) — inspection visuelle recommandée",
        "conclusif": False,
        "details": details,
    }


# ════════════════════════════════════════
# RÉPARTITEUR — extensible à d'autres marques
# ════════════════════════════════════════

# Marques disposant d'un détecteur de signature dédié
BRAND_DETECTORS = {
    "Tommy Hilfiger": tommy_hilfiger_signature,
}


def has_brand_detector(brand):
    """Indique si la marque dispose d'un détecteur de signature."""
    return brand in BRAND_DETECTORS


def brand_signature_check(img, brand):
    """
    Lance le détecteur de signature correspondant à la marque.
    Renvoie None si la marque n'a pas de détecteur dédié.
    """
    detector = BRAND_DETECTORS.get(brand)
    return detector(img) if detector else None
