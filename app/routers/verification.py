"""
Router Vérification IA — Pipeline de détection de contrefaçon
═════════════════════════════════════════════════════════════

Pipeline en 5 étapes appliquant une ANALYSE D'IMAGE RÉELLE sur les
photos uploadées (module app.services.cv_analysis).

Étape 1 — Prétraitement & contrôle qualité   → variance Laplacien, contraste
Étape 2 — Score global (features image)       → netteté + contraste + contours
Étape 3 — Analyse multi-têtes                  → logo, coutures, texture GLCM, étiquette
Étape 4 — Règles métier                        → cohérence prix, pHash anti-doublon
Étape 5 — Score pondéré + routing              → auto / expert / rejet

Les étapes 1, 3 (coutures, texture, étiquette) et 4 sont des analyses
réelles et définitives. Les étapes 2 et 3a (logo) calculent un proxy
déterministe basé sur les features de l'image ; en production elles sont
remplacées par EfficientNet-B4 et un Siamese Network entraînés.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.schemas import VerificationRequest
from app.models.models import Article, Verification, VerificationStatus, ArticlePhoto
from app.services import cv_analysis as cv
from app.services import invoice_analysis as inv
from app.ml import predict as ml_predict
import time

router = APIRouter()

# Pondérations du score final (somme = 1.0)
# La facture (preuve d'achat) pèse 0.20 — c'est un signal de confiance fort.
WEIGHTS = {"cnn": 0.22, "logo": 0.16, "stitching": 0.12,
           "texture": 0.12, "label": 0.08, "rules": 0.10, "invoice": 0.20}


@router.post("/launch", status_code=201)
def launch_verification(request: VerificationRequest, db: Session = Depends(get_db)):
    """Lance le pipeline de vérification IA sur un article."""
    article = db.query(Article).filter(Article.id == request.article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")

    photos = db.query(ArticlePhoto).filter(ArticlePhoto.article_id == article.id).all()
    if len(photos) < 3:
        raise HTTPException(status_code=400, detail="Minimum 3 photos requises pour la vérification")

    start = time.time()

    # Décodage des images réellement uploadées
    images = {}
    for p in photos:
        img = cv.decode_image(p.image_data) if p.image_data else None
        if img is not None:
            images[p.photo_type] = img

    has_real_images = len(images) > 0

    # ═══ ÉTAPE 1 — PRÉTRAITEMENT & CONTRÔLE QUALITÉ ═══
    quality_reports = {}
    if has_real_images:
        for ptype, img in images.items():
            quality_reports[ptype] = cv.quality_check(img)
        any_passed = any(q["passed"] for q in quality_reports.values())
        if not any_passed:
            raise HTTPException(
                status_code=400,
                detail="Photos de qualité insuffisante (flou, sous-exposition ou résolution trop faible)"
            )

    # Image de référence pour chaque tête d'analyse
    overview = images.get("overview") or _first(images)
    logo_img = images.get("logo") or overview
    stitch_img = images.get("stitching") or overview
    label_img = images.get("label") or overview

    # ═══ ÉTAPE 2 — SCORE GLOBAL (modèle ML entraîné) ═══
    # Le score global est désormais produit par le Random Forest entraîné
    # (app.ml). Si le modèle n'est pas disponible, repli heuristique.
    if overview:
        ml_result = ml_predict.predict_authenticity(overview)
        score_cnn = ml_result["score"]
    else:
        ml_result = {"score": _fallback("cnn", article), "source": "fallback_metadata",
                     "model_version": "fallback", "features": {}}
        score_cnn = ml_result["score"]

    # ═══ ÉTAPE 3 — ANALYSE MULTI-TÊTES ═══
    score_logo = cv.logo_score(logo_img) if logo_img else _fallback("logo", article)
    score_stitching = cv.stitching_score(stitch_img) if stitch_img else _fallback("stitching", article)
    if overview:
        score_texture, glcm = cv.texture_score(overview)
    else:
        score_texture, glcm = _fallback("texture", article), {}
    score_label = cv.label_score(label_img) if label_img else _fallback("label", article)

    # ═══ ÉTAPE 4 — RÈGLES MÉTIER ═══
    score_rules, rules_detail = _business_rules(article, images, db)

    # ═══ ÉTAPE 4bis — VÉRIFICATION DE LA FACTURE D'ACHAT ═══
    # OCR : en production, le texte est extrait par Google Vision API.
    # On accepte un texte optionnel transmis dans la requête.
    ocr_text = getattr(request, "invoice_ocr_text", None)
    if article.invoice_data:
        invoice_report = inv.verify_invoice(article.invoice_data, article, ocr_text)
        score_invoice = invoice_report["score"]
    else:
        invoice_report = {
            "score": 0.5,
            "verdict": "Aucune facture fournie — preuve d'achat manquante",
            "controles": {},
        }
        score_invoice = 0.5

    # ═══ ÉTAPE 5 — SCORE PONDÉRÉ ═══
    score_global = (
        WEIGHTS["cnn"] * score_cnn +
        WEIGHTS["logo"] * score_logo +
        WEIGHTS["stitching"] * score_stitching +
        WEIGHTS["texture"] * score_texture +
        WEIGHTS["label"] * score_label +
        WEIGHTS["rules"] * score_rules +
        WEIGHTS["invoice"] * score_invoice
    )

    processing_time = int((time.time() - start) * 1000)

    # Routing 3 niveaux
    if score_global >= 0.85:
        status = VerificationStatus.VERIFIED
    elif score_global >= 0.40:
        status = VerificationStatus.REVIEW
    else:
        status = VerificationStatus.REJECTED

    verification = Verification(
        article_id=article.id,
        status=status,
        score_global=round(score_global, 4),
        score_logo=round(score_logo, 4),
        score_stitching=round(score_stitching, 4),
        score_texture=round(score_texture, 4),
        score_label=round(score_label, 4),
        score_rules=round(score_rules, 4),
        score_invoice=round(score_invoice, 4),
        pipeline_version="v0.4-ml" if has_real_images else "v0.1-poc",
        processing_time_ms=processing_time,
        gradcam_path=f"gs://luxloc-gradcam/{article.id}/report.png",
    )
    db.add(verification)
    if status == VerificationStatus.VERIFIED:
        article.is_available = True
    db.commit()
    db.refresh(verification)

    return {
        "verification_id": verification.id,
        "article_id": article.id,
        "status": status.value,
        "verdict": _verdict_label(status),
        "scores": {
            "global": verification.score_global,
            "cnn": round(score_cnn, 4),
            "logo": verification.score_logo,
            "stitching": verification.score_stitching,
            "texture": verification.score_texture,
            "label": verification.score_label,
            "rules": verification.score_rules,
            "invoice": verification.score_invoice,
        },
        "analysis": {
            "mode": "computer_vision_reelle" if has_real_images else "fallback_metadata",
            "images_analysees": list(images.keys()),
            "controle_qualite": quality_reports,
            "modele_ml": {
                "score_authenticite": ml_result["score"],
                "source": ml_result["source"],
                "version": ml_result["model_version"],
                "features_extraites": ml_result["features"],
            },
            "glcm_texture": {k: round(v, 4) for k, v in glcm.items()} if glcm else {},
            "regles_metier": rules_detail,
            "verification_facture": invoice_report,
        },
        "pipeline": {
            "version": "v0.4-ml" if has_real_images else "v0.1-poc",
            "processing_time_ms": processing_time,
            "steps_completed": ["preprocessing", "global_score", "multi_head_analysis",
                                "business_rules", "invoice_verification", "weighted_scoring"],
            "weights": WEIGHTS,
        },
        "gradcam_url": verification.gradcam_path,
    }


@router.get("/model/info")
def get_model_info():
    """
    Renvoie la fiche du modèle ML de détection de contrefaçon :
    métriques d'évaluation, features utilisées, importance des features.
    Utile pour la transparence et la présentation du modèle.
    """
    info = ml_predict.model_info()
    if not info:
        return {
            "model_disponible": False,
            "message": "Modèle non entraîné — exécuter : python -m app.ml.train",
        }
    return {
        "model_disponible": True,
        "modele": info["model"],
        "jeu_de_donnees": {
            "taille_totale": info["dataset_size"],
            "entrainement": info["train_size"],
            "test": info["test_size"],
        },
        "metriques": {
            "accuracy": info["accuracy"],
            "precision": info["precision"],
            "recall": info["recall"],
            "f1_score": info["f1_score"],
            "f1_validation_croisee": f"{info['cv_f1_mean']} (+/- {info['cv_f1_std']})",
        },
        "matrice_confusion": info["confusion_matrix"],
        "importance_features": info["feature_importances"],
    }


@router.get("/{article_id}")
def get_verification(article_id: int, db: Session = Depends(get_db)):
    """Récupérer le résultat de vérification d'un article."""
    verif = (db.query(Verification)
             .filter(Verification.article_id == article_id)
             .order_by(Verification.id.desc())
             .first())
    if not verif:
        raise HTTPException(status_code=404, detail="Aucune vérification trouvée")
    return {
        "status": verif.status.value,
        "scores": {
            "global": verif.score_global,
            "logo": verif.score_logo,
            "stitching": verif.score_stitching,
            "texture": verif.score_texture,
            "label": verif.score_label,
            "rules": verif.score_rules,
            "invoice": verif.score_invoice,
        },
        "pipeline_version": verif.pipeline_version,
        "processing_time_ms": verif.processing_time_ms,
        "created_at": verif.created_at,
    }


# ════════════════════════════════════════
# ÉTAPE 4 — RÈGLES MÉTIER (analyse réelle)
# ════════════════════════════════════════

def _business_rules(article, images, db):
    """
    Règles métier réelles :
    - cohérence du ratio prix location / prix neuf
    - détection de doublon via pHash (image déjà soumise)
    - historique du propriétaire (taux de rejet passé)
    """
    score = 1.0
    detail = {}

    # 1. Cohérence prix
    if article.retail_price and article.price_per_day:
        ratio = article.price_per_day / article.retail_price
        detail["ratio_prix"] = round(ratio, 4)
        if ratio < 0.005:
            score -= 0.35
            detail["alerte_prix"] = "Prix de location anormalement bas"
        elif ratio > 0.15:
            score -= 0.25
            detail["alerte_prix"] = "Prix de location anormalement élevé"
        else:
            detail["alerte_prix"] = "Cohérent"
    else:
        detail["alerte_prix"] = "Prix non renseigné"
        score -= 0.10

    # 2. Détection de doublon par pHash
    overview = images.get("overview") or _first(images)
    if overview is not None:
        phash = cv.perceptual_hash(overview)
        detail["phash"] = phash
        others = db.query(ArticlePhoto).filter(
            ArticlePhoto.photo_type == "overview",
            ArticlePhoto.article_id != article.id,
            ArticlePhoto.image_data.isnot(None),
        ).all()
        duplicate = False
        for o in others:
            other_img = cv.decode_image(o.image_data)
            if other_img is None:
                continue
            if cv.hamming_distance(phash, cv.perceptual_hash(other_img)) <= 5:
                duplicate = True
                break
        if duplicate:
            score -= 0.40
            detail["alerte_doublon"] = "Image déjà présente sur la plateforme"
        else:
            detail["alerte_doublon"] = "Aucun doublon détecté"

    # 3. Historique du propriétaire
    if article.owner_id:
        past = (db.query(Verification)
                .join(Article, Verification.article_id == Article.id)
                .filter(Article.owner_id == article.owner_id)
                .all())
        if past:
            rejets = sum(1 for v in past if v.status == VerificationStatus.REJECTED)
            taux_rejet = rejets / len(past)
            detail["taux_rejet_proprietaire"] = round(taux_rejet, 2)
            if taux_rejet > 0.3:
                score -= 0.15
                detail["alerte_historique"] = "Historique de rejets élevé"
        else:
            detail["taux_rejet_proprietaire"] = 0.0

    return max(0.0, min(score, 1.0)), detail


# ════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════

def _first(images):
    """Renvoie la première image disponible, ou None."""
    for v in images.values():
        return v
    return None


def _fallback(component, article):
    """
    Score de repli quand aucune image n'est disponible (compatibilité
    avec les anciens articles sans image stockée). Déterministe.
    """
    premium = {"Louis Vuitton", "Hermès", "Chanel", "Gucci", "Rolex", "Cartier", "Dior"}
    base = 0.80 if article.brand in premium else 0.72
    offsets = {"cnn": 0.05, "logo": 0.04, "stitching": 0.02,
               "texture": 0.03, "label": -0.02}
    return max(0.0, min(base + offsets.get(component, 0.0), 1.0))


def _verdict_label(status):
    return {
        VerificationStatus.VERIFIED: "Article authentifié — Auto-validé",
        VerificationStatus.REVIEW: "Incertain — Envoyé à un expert humain",
        VerificationStatus.REJECTED: "Suspect — Article rejeté",
    }.get(status, "Inconnu")
