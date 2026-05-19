"""
LUXLOC — Module d'analyse de facture
═══════════════════════════════════════

Vérification RÉELLE de la facture d'achat uploadée par le propriétaire.
Utilise Pillow + numpy uniquement.

Le pipeline de vérification de facture comporte 4 contrôles :

  1. Contrôle qualité       → l'image est-elle lisible (netteté, résolution) ?
  2. Error Level Analysis   → la facture a-t-elle été retouchée (Photoshop) ?
  3. Validation de format   → le n° de facture respecte-t-il le format de la marque ?
  4. Cohérence article      → la marque/le prix de l'article correspondent-ils ?

Les contrôles 1 et 2 sont des analyses d'image réelles et définitives.
Le contrôle 3 s'appuie sur un texte OCR : en production, l'extraction est
faite par Google Vision API ; ici la fonction accepte un texte fourni
(ou renvoie un score neutre si aucun texte n'est disponible).
Le contrôle 4 est une règle métier réelle.
"""
import base64
import io
import re
import math
from PIL import Image, ImageChops
import numpy as np


# ════════════════════════════════════════
# DÉCODAGE
# ════════════════════════════════════════

def decode_image(data_uri_or_b64):
    """Décode une image base64 / data-URI en objet PIL. Renvoie None si échec."""
    if not data_uri_or_b64:
        return None
    try:
        raw = data_uri_or_b64
        if "," in raw and raw.strip().startswith("data:"):
            raw = raw.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(raw)))
        return img.convert("RGB")
    except Exception:
        return None


# ════════════════════════════════════════
# CONTRÔLE 1 — QUALITÉ DE LA FACTURE (réel)
# ════════════════════════════════════════

def quality_check(img):
    """
    Vérifie que la facture est exploitable :
    - résolution suffisante pour lire le texte
    - image nette (variance du Laplacien)
    - contraste suffisant (texte vs fond)
    """
    w, h = img.size
    gray = np.asarray(img.convert("L"), dtype=np.float64)

    # Netteté — variance du Laplacien
    lap = (
        -4 * gray
        + np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
        + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
    )
    sharpness = float(lap.var())

    # Contraste — écart-type des niveaux de gris
    contrast = float(gray.std())

    resolution_ok = w >= 600 and h >= 600
    sharpness_ok = sharpness >= 80.0
    contrast_ok = contrast >= 35.0

    passed = resolution_ok and sharpness_ok and contrast_ok
    score = (
        (0.34 if resolution_ok else 0.0)
        + (0.33 if sharpness_ok else 0.0)
        + (0.33 if contrast_ok else 0.0)
    )

    return {
        "passed": passed,
        "score": round(score, 3),
        "resolution": f"{w}x{h}",
        "resolution_ok": resolution_ok,
        "nettete": round(sharpness, 1),
        "nettete_ok": sharpness_ok,
        "contraste": round(contrast, 1),
        "contraste_ok": contrast_ok,
    }


# ════════════════════════════════════════
# CONTRÔLE 2 — ERROR LEVEL ANALYSIS (réel)
# ════════════════════════════════════════

def error_level_analysis(img, quality=90):
    """
    Error Level Analysis (ELA) — indicateur de cohérence de compression.

    Principe : on recompresse l'image en JPEG à qualité fixe et on mesure
    la différence avec l'originale. Une image homogène (prise/scannée en
    une fois) présente un niveau d'erreur uniforme. Une retouche grossière
    (zone collée depuis une autre source) peut créer une rupture locale.

    NOTE — portée réelle : l'ELA est un indicateur, pas une preuve. Il est
    fiable sur des photographies réelles de factures ; sur un document
    purement numérique il est moins discriminant. Il contribue donc
    modérément au score, en complément du contrôle de cohérence du prix.

    Renvoie un score 0-1 (1 = compression homogène) et des métriques.
    """
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(img, recompressed)
    magnitude = np.asarray(diff, dtype=np.float64).mean(axis=2)

    mean_err = float(magnitude.mean())
    max_err = float(magnitude.max())
    std_err = float(magnitude.std())

    # Découpage en grille 16x16 et analyse de l'homogénéité.
    h, w = magnitude.shape
    GRID = 16
    bh, bw = max(1, h // GRID), max(1, w // GRID)
    block_means = []
    for by in range(0, h - bh + 1, bh):
        for bx in range(0, w - bw + 1, bw):
            block_means.append(float(magnitude[by:by + bh, bx:bx + bw].mean()))
    block_means = np.array(block_means) if block_means else np.array([mean_err])

    median_block = float(np.median(block_means))
    p95_block = float(np.percentile(block_means, 95))

    # Indice d'homogénéité : rapport entre le 95e percentile et la médiane
    # des blocs porteurs de contenu. Un document homogène a un rapport
    # modéré ; une rupture franche le fait grimper.
    content_blocks = block_means[block_means > 0.02]
    if len(content_blocks) >= 4:
        ref = float(np.median(content_blocks))
        hotspot_ratio = p95_block / max(ref, 0.01)
    else:
        hotspot_ratio = 1.0

    # Score : élevé par défaut, pénalisé seulement si la dispersion est
    # franchement anormale (rupture de compression marquée).
    score = 1.0
    if hotspot_ratio > 8.0:
        score -= 0.30
    elif hotspot_ratio > 5.0:
        score -= 0.15
    if std_err > 25.0:
        score -= 0.15
    score = max(0.0, min(score, 1.0))

    if score >= 0.9:
        verdict = "Compression homogène — aucune anomalie"
    elif score >= 0.7:
        verdict = "Légère hétérogénéité — contrôle complémentaire conseillé"
    else:
        verdict = "Forte hétérogénéité de compression — facture à examiner"

    return {
        "score": round(score, 3),
        "erreur_moyenne": round(mean_err, 3),
        "erreur_max": round(max_err, 3),
        "dispersion": round(std_err, 3),
        "indice_heterogeneite": round(hotspot_ratio, 2),
        "verdict": verdict,
    }


# ════════════════════════════════════════
# CONTRÔLE 3 — VALIDATION DE FORMAT (réel sur texte fourni)
# ════════════════════════════════════════

# Formats de numéro de facture / série par marque (regex réelles, simplifiées).
BRAND_INVOICE_PATTERNS = {
    "Louis Vuitton": r"\b[A-Z]{2}\d{4}\b",          # ex: SD4012
    "Hermès":        r"\b[A-Z]\d{6,8}\b",            # ex: T1234567
    "Chanel":        r"\b\d{7,8}\b",                 # ex: 28912345
    "Gucci":         r"\b\d{6}[\s\-]?\d{6}\b",       # ex: 526415-001998
    "Rolex":         r"\b[A-Z]?\d{6,8}\b",           # ex: 4012345
    "Cartier":       r"\b[A-Z]{2}\d{6}\b",           # ex: CR123456
    "Dior":          r"\b\d{2}-[A-Z]{2}-\d{4}\b",    # ex: 05-BO-0145
    "Prada":         r"\b\d[A-Z]{2}\d{3}\b",         # ex: 1BA204
    "Burberry":      r"\b[A-Z]{2}\d{6,8}\b",
    "Tommy Hilfiger": r"\b[A-Z]{2}\d{6,9}\b",   # ex: TH12345678
}

# Boutiques officielles connues (extrait — en production : store locator scrapé).
OFFICIAL_BOUTIQUES = {
    "louis vuitton", "hermès", "hermes", "chanel", "gucci", "rolex",
    "cartier", "dior", "prada", "burberry", "galeries lafayette",
    "le bon marché", "printemps", "24 sèvres", "fauré le page",
    "tommy hilfiger", "tommy",
}


def format_validation(ocr_text, brand):
    """
    Valide le format du numéro de facture selon la marque, et vérifie
    que la boutique citée appartient au réseau officiel.

    `ocr_text` est le texte extrait de la facture (Google Vision API en
    production). Si vide, renvoie un score neutre signalant l'absence d'OCR.
    """
    if not ocr_text or not ocr_text.strip():
        return {
            "score": 0.5,
            "ocr_disponible": False,
            "note": "Aucun texte OCR fourni — extraction Google Vision requise en production",
        }

    text_low = ocr_text.lower()
    detail = {"ocr_disponible": True}
    score = 0.0

    # 1. Numéro de facture au bon format
    pattern = BRAND_INVOICE_PATTERNS.get(brand)
    if pattern:
        match = re.search(pattern, ocr_text)
        if match:
            score += 0.45
            detail["numero_facture"] = match.group(0)
            detail["format_numero"] = "Conforme au format " + brand
        else:
            detail["format_numero"] = "Numéro absent ou format non conforme"
    else:
        score += 0.20
        detail["format_numero"] = "Marque sans format de référence enregistré"

    # 2. Marque mentionnée sur la facture
    if brand.lower() in text_low:
        score += 0.25
        detail["marque_presente"] = True
    else:
        detail["marque_presente"] = False

    # 3. Boutique officielle citée
    boutique = next((b for b in OFFICIAL_BOUTIQUES if b in text_low), None)
    if boutique:
        score += 0.30
        detail["boutique"] = boutique.title()
        detail["boutique_officielle"] = True
    else:
        detail["boutique_officielle"] = False

    detail["score"] = round(min(score, 1.0), 3)
    return detail


# ════════════════════════════════════════
# CONTRÔLE 4 — COHÉRENCE ARTICLE ↔ FACTURE (réel)
# ════════════════════════════════════════

def article_consistency(article, ocr_text):
    """
    Vérifie que l'article déclaré correspond à la facture :
    - le modèle déclaré apparaît-il dans le texte de la facture ?
    - le prix neuf déclaré est-il cohérent avec un montant de la facture ?
    """
    detail = {}
    score = 1.0

    if not ocr_text or not ocr_text.strip():
        return {"score": 0.5, "note": "Cross-référencement impossible sans OCR"}

    text_low = ocr_text.lower()

    # 1. Modèle présent
    if article.model and article.model.lower() in text_low:
        detail["modele_present"] = True
    else:
        detail["modele_present"] = False
        score -= 0.30

    # 2. Cohérence du prix : on cherche les montants en euros bien formés
    #    (groupes de milliers séparés par espace/point, décimale optionnelle).
    #    Ex. capturés : "1 250", "1 250,00", "6 900,00", "1250.00"
    montants = re.findall(
        r"(\d{1,3}(?:[ .]\d{3})*(?:,\d{2})?)\s*(?:€|eur|euros)",
        text_low,
    )
    prix_factures = []
    for m in montants:
        cleaned = m.replace(" ", "").replace(".", "").replace(",", ".")
        try:
            val = float(cleaned)
            if 50 <= val <= 500000:  # borne raisonnable pour un article de luxe
                prix_factures.append(val)
        except ValueError:
            pass

    if prix_factures and article.retail_price:
        plus_proche = min(prix_factures, key=lambda p: abs(p - article.retail_price))
        ecart = abs(plus_proche - article.retail_price) / article.retail_price
        detail["prix_facture_detecte"] = plus_proche
        detail["ecart_prix"] = round(ecart, 3)
        if ecart <= 0.20:
            detail["coherence_prix"] = "Cohérent"
        elif ecart <= 0.50:
            detail["coherence_prix"] = "Écart de prix modéré"
            score -= 0.30
        elif ecart <= 1.0:
            detail["coherence_prix"] = "Écart de prix important"
            score -= 0.55
        else:
            detail["coherence_prix"] = "Prix incohérent — falsification probable"
            score -= 0.90
    else:
        detail["coherence_prix"] = "Aucun montant exploitable sur la facture"
        score -= 0.10

    detail["score"] = round(max(0.0, min(score, 1.0)), 3)
    return detail


# ════════════════════════════════════════
# PIPELINE COMPLET DE VÉRIFICATION FACTURE
# ════════════════════════════════════════

# Pondérations des 4 contrôles (somme = 1.0)
INVOICE_WEIGHTS = {"quality": 0.15, "ela": 0.40, "format": 0.25, "consistency": 0.20}


def verify_invoice(invoice_b64, article, ocr_text=None):
    """
    Exécute le pipeline complet de vérification de facture et renvoie
    un score global 0-1 + le détail de chaque contrôle.

    `ocr_text` est optionnel : fourni par Google Vision API en production.
    """
    img = decode_image(invoice_b64)
    if img is None:
        return {
            "score": 0.0,
            "verdict": "Facture illisible ou absente",
            "controles": {},
        }

    quality = quality_check(img)
    ela = error_level_analysis(img)
    fmt = format_validation(ocr_text, article.brand)
    consistency = article_consistency(article, ocr_text)

    score = (
        INVOICE_WEIGHTS["quality"] * quality["score"]
        + INVOICE_WEIGHTS["ela"] * ela["score"]
        + INVOICE_WEIGHTS["format"] * fmt["score"]
        + INVOICE_WEIGHTS["consistency"] * consistency["score"]
    )

    # Veto : une incohérence de prix flagrante (montant facturé sans rapport
    # avec l'article déclaré) est un signal de falsification fort. Le score
    # global de la facture est plafonné, quels que soient les autres contrôles.
    if consistency.get("coherence_prix", "").startswith("Prix incohérent"):
        score = min(score, 0.35)

    score = round(max(0.0, min(score, 1.0)), 4)

    if score >= 0.80:
        verdict = "Facture authentique — preuve d'achat validée"
    elif score >= 0.50:
        verdict = "Facture incertaine — vérification humaine recommandée"
    else:
        verdict = "Facture suspecte — preuve d'achat non validée"

    return {
        "score": score,
        "verdict": verdict,
        "poids": INVOICE_WEIGHTS,
        "controles": {
            "qualite": quality,
            "error_level_analysis": ela,
            "validation_format": fmt,
            "coherence_article": consistency,
        },
    }
