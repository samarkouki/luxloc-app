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
from app.schemas.schemas import VerificationRequest, ExpertRejectRequest
from app.models.models import Article, Verification, VerificationStatus, ArticlePhoto, User
from app.dependencies import require_admin, log_admin_action
from app.services import cv_analysis as cv
from app.services import invoice_analysis as inv
from app.services import brand_signature as brand_sig_mod
from app.ml import predict as ml_predict
from app.ml import authenticator as efficientnet
import threading
import time
import traceback

router = APIRouter()

# Pondérations du score final (somme = 1.0)
# La facture (preuve d'achat) pèse 0.20 — c'est un signal de confiance fort.
WEIGHTS = {"cnn": 0.22, "logo": 0.16, "stitching": 0.12,
           "texture": 0.12, "label": 0.08, "rules": 0.10, "invoice": 0.20}


# ════════════════════════════════════════
# SUIVI DE PROGRESSION
# ════════════════════════════════════════
# La vérification tient dans une seule requête, qui ne répond qu'à la fin
# (jusqu'à une minute au premier chargement du modèle). Pendant ce temps,
# l'écran du propriétaire interroge GET /{article_id}/progress pour afficher
# l'étape réellement en cours. Le suivi vit en mémoire du process : il
# suffit avec un seul process uvicorn (le cas sur Railway) et disparaît à
# un redémarrage — le résultat final, lui, est en base.
# LIMITE : avec plusieurs process (uvicorn --workers N, ou plusieurs
# instances Railway), l'interrogation peut tomber sur un process qui n'a
# pas lancé l'analyse : l'écran ne verrait alors aucune progression et
# l'espace propriétaire n'afficherait pas « analyse en cours ». Avant de
# passer à plusieurs process, déplacer ce suivi en base ou dans Redis.
ETAPES = [
    ("qualite", "Contrôle qualité des photos"),
    ("authenticite", "Score d'authenticité"),
    ("logo", "Logo et monogramme"),
    ("coutures", "Coutures et finitions"),
    ("texture", "Texture des matériaux"),
    ("etiquette", "Étiquette et numéro de série"),
    ("regles", "Règles métier"),
    ("facture", "Facture d'achat"),
]
_CLES_ETAPES = [cle for cle, _ in ETAPES]
_progress = {}
_progress_lock = threading.Lock()
_PROGRESS_TTL = 15 * 60


def _progress_set(article_id, **fields):
    with _progress_lock:
        now = time.time()
        for aid in [a for a, p in _progress.items() if now - p["maj"] > _PROGRESS_TTL]:
            del _progress[aid]
        entry = _progress.setdefault(article_id, {})
        entry.update(fields, maj=now)


def _etape(article_id, cle, **extra):
    _progress_set(article_id, etape=cle, **extra)


def analyse_en_cours(article_id):
    """Vrai si une vérification de cet article est en train de tourner."""
    with _progress_lock:
        p = _progress.get(article_id)
        return bool(p and p.get("etat") == "en_cours")


@router.post("/launch", status_code=201)
def launch_verification(request: VerificationRequest, db: Session = Depends(get_db)):
    """Lance le pipeline de vérification IA sur un article."""
    aid = request.article_id
    _progress_set(aid, etat="en_cours", etape=_CLES_ETAPES[0], debut=time.time(),
                  chargement_modele=False, resultat=None, erreur=None)
    try:
        result = _run_verification(request, db)
    except HTTPException as e:
        _progress_set(aid, etat="echec", erreur=str(e.detail))
        raise
    except Exception:
        _progress_set(aid, etat="echec", erreur="serveur")
        raise
    _progress_set(aid, etat="termine", etape=None, resultat=result)
    return result


@router.get("/{article_id}/progress")
def verification_progress(article_id: int):
    """
    Avancement réel de la vérification d'un article : étape en cours, étapes
    terminées, et le résultat complet une fois l'analyse finie (le même que
    celui renvoyé par POST /launch).
    """
    with _progress_lock:
        p = dict(_progress.get(article_id) or {})
    if not p:
        return {"etat": "inconnu"}
    courante = p.get("etape")
    index = _CLES_ETAPES.index(courante) if courante in _CLES_ETAPES else len(_CLES_ETAPES)
    if p["etat"] == "termine":
        index = len(_CLES_ETAPES)
    etapes = [{
        "cle": cle,
        "libelle": libelle,
        "statut": "termine" if i < index else ("en_cours" if i == index and p["etat"] == "en_cours" else "en_attente"),
    } for i, (cle, libelle) in enumerate(ETAPES)]
    return {
        "etat": p["etat"],
        "etapes": etapes,
        "chargement_modele": bool(p.get("chargement_modele")) and courante == "authenticite",
        "duree_s": round(time.time() - p.get("debut", time.time()), 1),
        "erreur": p.get("erreur"),
        "resultat": p.get("resultat") if p["etat"] == "termine" else None,
    }


def _run_verification(request: VerificationRequest, db: Session):
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
    # On ne refuse plus un article pour des conditions de prise de vue
    # imparfaites (lumière, fond, léger flou) : l'analyse va au bout, et si
    # aucune photo n'est de bonne qualité, l'article part en examen expert
    # (étape 5). Seule une résolution inexploitable (< 200 px) bloque.
    quality_reports = {}
    qualite_insuffisante = False
    if has_real_images:
        for ptype, img in images.items():
            quality_reports[ptype] = cv.quality_check(img)
        if not any(q["resolution_ok"] for q in quality_reports.values()):
            raise HTTPException(
                status_code=400,
                detail="Photos trop petites pour être analysées (moins de 200 pixels de côté)"
            )
        qualite_insuffisante = not any(q["passed"] for q in quality_reports.values())

    # Image de référence pour chaque tête d'analyse
    overview = images.get("overview") or _first(images)
    logo_img = images.get("logo") or overview
    stitch_img = images.get("stitching") or overview
    label_img = images.get("label") or overview

    # ═══ ÉTAPE 2 — SCORE GLOBAL (authenticité) ═══
    # Le score vient d'EfficientNet-B0 (luxury_authenticator_v1) quand le
    # modèle a pu être chargé ; sinon repli silencieux sur le Random Forest
    # historique (app.ml), qui a lui-même un repli heuristique — l'un des
    # trois niveaux répond toujours. EfficientNet analyse les quatre photos
    # (pas seulement l'aperçu) : le score retenu est la probabilité la PLUS
    # BASSE parmi les photos concluantes (confiance >= 0,80) — une seule vue
    # suspecte doit faire baisser le score. Une photo non concluante ne
    # compte pas comme preuve d'authenticité et est exclue du calcul ; si
    # toutes le sont, le routage vers l'expert est forcé à l'étape 5, quel
    # que soit le score pondéré.
    toutes_photos_non_concluantes = False
    ml_result = None
    erreur_efficientnet = None
    detail_par_photo = None
    # Premier passage depuis le démarrage : le chargement du modèle prend
    # l'essentiel du temps, l'écran du propriétaire le signale.
    _etape(article.id, "authenticite", chargement_modele=(efficientnet.status() != "actif"))
    if overview and efficientnet.is_available():
        # Une panne pendant la prédiction (et pas seulement au chargement)
        # ne doit jamais faire échouer la vérification : on retombe sur le
        # Random Forest et la cause part dans les journaux.
        try:
            detail_par_photo = {}
            for ptype in ("overview", "logo", "stitching", "label"):
                img = images.get(ptype)
                if img is not None:
                    detail_par_photo[ptype] = efficientnet.predict_one(img)
        except Exception as e:
            erreur_efficientnet = f"{type(e).__name__}: {e}"
            print(f"[efficientnet] échec de l'analyse, repli sur le Random Forest — {erreur_efficientnet}")
            traceback.print_exc()
            detail_par_photo = None

    if detail_par_photo is not None:
        concluantes = {k: v for k, v in detail_par_photo.items() if v["concluant"]}
        non_concluantes = [k for k in detail_par_photo if k not in concluantes]

        if concluantes:
            pire = min(concluantes, key=lambda k: concluantes[k]["score"])
            score_cnn = concluantes[pire]["score"]
        else:
            # Aucune photo concluante : score neutre par défaut, le
            # routage vers l'expert est forcé plus bas (étape 5).
            score_cnn = 0.5
            toutes_photos_non_concluantes = True

        ml_result = {
            "score": round(score_cnn, 4),
            "source": "efficientnet_b0",
            "model_version": efficientnet.MODEL_VERSION,
            "features": {},
            "detail_par_photo": detail_par_photo,
            "moyenne_photos": round(
                sum(v["score"] for v in detail_par_photo.values()) / len(detail_par_photo), 4
            ) if detail_par_photo else None,
            "photos_non_concluantes": non_concluantes,
        }
    elif overview:
        ml_result = ml_predict.predict_authenticity(overview)
        score_cnn = ml_result["score"]
        if erreur_efficientnet:
            ml_result["erreur_efficientnet"] = erreur_efficientnet
    else:
        ml_result = {"score": _fallback("cnn", article), "source": "fallback_metadata",
                     "model_version": "fallback", "features": {}}
        score_cnn = ml_result["score"]

    # ═══ ÉTAPE 3 — ANALYSE MULTI-TÊTES ═══
    _etape(article.id, "logo", chargement_modele=False)
    score_logo = cv.logo_score(logo_img) if logo_img else _fallback("logo", article)

    # Étape 3a-bis — Signature de marque (ex. Tommy Hilfiger)
    # Si la marque dispose d'un détecteur dédié, on analyse la signature
    # couleur du logo et on la combine au score logo générique.
    brand_sig = None
    if logo_img is not None and brand_sig_mod.has_brand_detector(article.brand):
        brand_sig = brand_sig_mod.brand_signature_check(logo_img, article.brand)
        # On ne combine la signature au score logo que si elle est concluante
        # (variante identifiée). Sinon elle reste purement informative.
        if brand_sig and brand_sig.get("conclusif"):
            score_logo = 0.5 * score_logo + 0.5 * brand_sig["score"]

    _etape(article.id, "coutures")
    score_stitching = cv.stitching_score(stitch_img) if stitch_img else _fallback("stitching", article)
    _etape(article.id, "texture")
    if overview:
        score_texture, glcm = cv.texture_score(overview)
    else:
        score_texture, glcm = _fallback("texture", article), {}
    _etape(article.id, "etiquette")
    score_label = cv.label_score(label_img) if label_img else _fallback("label", article)

    # ═══ ÉTAPE 4 — RÈGLES MÉTIER ═══
    _etape(article.id, "regles")
    score_rules, rules_detail = _business_rules(article, images, db)

    # ═══ ÉTAPE 4bis — VÉRIFICATION DE LA FACTURE D'ACHAT ═══
    # OCR : en production, le texte est extrait par Google Vision API.
    # On accepte un texte optionnel transmis dans la requête.
    _etape(article.id, "facture")
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

    # Routing 3 niveaux — si aucune photo n'a été concluante pour
    # EfficientNet, on ne dispose d'aucune preuve visuelle exploitable :
    # l'article part systématiquement en examen expert, jamais en
    # validation automatique (ni en rejet automatique, faute de preuve).
    # Même règle quand aucune photo n'a passé le contrôle qualité : les
    # scores visuels ne sont pas fiables, un expert tranche.
    if toutes_photos_non_concluantes or qualite_insuffisante:
        status = VerificationStatus.REVIEW
    elif score_global >= 0.85:
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
            "qualite_photos_insuffisante": qualite_insuffisante,
            "modele_ml": {
                "score_authenticite": ml_result["score"],
                "source": ml_result["source"],
                "version": ml_result["model_version"],
                "features_extraites": ml_result["features"],
                "detail_par_photo": ml_result.get("detail_par_photo", {}),
                "moyenne_photos": ml_result.get("moyenne_photos"),
                "photos_non_concluantes": ml_result.get("photos_non_concluantes", []),
                "toutes_photos_non_concluantes": toutes_photos_non_concluantes,
                "erreur_efficientnet": erreur_efficientnet,
            },
            "glcm_texture": {k: round(v, 4) for k, v in glcm.items()} if glcm else {},
            "signature_marque": brand_sig,
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


@router.post("/{article_id}/expert-validate")
def expert_validate(article_id: int, admin: User = Depends(require_admin),
                     db: Session = Depends(get_db)):
    """
    Validation par un expert humain — étape de routing pour les articles
    en statut 'review' (score 0.40-0.84). L'expert tranche : l'article
    est validé, sa dernière vérification passe à VERIFIED et il devient
    disponible à la location.

    Réservé à un compte admin (voir app.dependencies.require_admin), et
    refusé si l'appelant est le propriétaire de l'article : un
    propriétaire ne doit jamais pouvoir valider sa propre pièce, cela
    viderait de sens le routage vers un expert humain.
    """
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")
    if article.owner_id == admin.id:
        raise HTTPException(status_code=403,
                            detail="Un propriétaire ne peut pas valider son propre article")

    verif = (db.query(Verification)
             .filter(Verification.article_id == article_id)
             .order_by(Verification.id.desc())
             .first())
    if not verif:
        raise HTTPException(status_code=400, detail="Aucune vérification à valider")
    if verif.status == VerificationStatus.REJECTED:
        raise HTTPException(status_code=400,
                            detail="Article rejeté — non validable par expert")

    verif.status = VerificationStatus.VERIFIED
    verif.reviewed_by = f"admin:{admin.id}"
    article.is_available = True
    log_admin_action(db, admin.id, "validate_article", "article", article.id,
                      f"« {article.title} » validé par un expert")
    db.commit()

    return {
        "message": "Article validé par l'expert",
        "article_id": article_id,
        "status": "verified",
        "is_available": True,
    }


@router.post("/{article_id}/expert-reject")
def expert_reject(article_id: int, payload: ExpertRejectRequest,
                   admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """
    Rejet motivé par un expert humain — pendant du expert-validate pour
    les articles en 'review' qui ne passent pas le contrôle. Le motif est
    obligatoire et conservé dans le journal d'administration.
    """
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")
    if article.owner_id == admin.id:
        raise HTTPException(status_code=403,
                            detail="Un propriétaire ne peut pas rejeter son propre article")

    verif = (db.query(Verification)
             .filter(Verification.article_id == article_id)
             .order_by(Verification.id.desc())
             .first())
    if not verif:
        raise HTTPException(status_code=400, detail="Aucune vérification à traiter")

    verif.status = VerificationStatus.REJECTED
    verif.reviewed_by = f"admin:{admin.id}"
    article.is_available = False
    log_admin_action(db, admin.id, "reject_article", "article", article.id, payload.motif)
    db.commit()

    return {
        "message": "Article rejeté par l'expert",
        "article_id": article_id,
        "status": "rejected",
        "motif": payload.motif,
    }


@router.post("/{article_id}/relaunch")
def relaunch_verification(article_id: int, admin: User = Depends(require_admin),
                           db: Session = Depends(get_db)):
    """Relance le pipeline de vérification IA sur un article déjà soumis."""
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")

    # photo_ids n'est en réalité pas exploité par launch_verification, qui
    # relit les photos réelles depuis la base à partir de l'article_id —
    # on ne le renseigne ici que pour satisfaire la validation du schéma.
    request = VerificationRequest(article_id=article_id, photo_ids=[1, 2, 3])
    result = launch_verification(request, db)
    log_admin_action(db, admin.id, "relaunch_verification", "article", article.id,
                      f"Nouveau statut : {result['status']}")
    db.commit()
    return result


@router.get("/model/info")
def get_model_info():
    """
    Fiche du modèle produisant le score d'authenticité de l'étape 2 —
    EfficientNet-B0, qui a remplacé le Random Forest (voir MODELES.md pour
    la comparaison complète). Lit metrics.json sans jamais déclencher le
    chargement du modèle lui-même : cette fiche reste consultable même
    avant la première vérification.
    """
    info = efficientnet.model_info()
    if not info:
        return {
            "model_disponible": False,
            "message": "Fiche modèle indisponible — metrics.json absent de luxury_authenticator_v1/models/.",
        }

    cm = info["confusion_matrix"]["matrix"]  # [[fake→fake, fake→real], [real→fake, real→real]]
    return {
        "model_disponible": True,
        "architecture": "EfficientNet-B0 (fine-tuné)",
        "version": efficientnet.MODEL_VERSION,
        "jeu_de_donnees": {
            "taille_test": info["test_images"],
        },
        "metriques": {
            "accuracy": info["accuracy"],
            "precision": info["precision"],
            "recall": info["recall"],
            "f1_score": info["f1_score"],
            "roc_auc": info["roc_auc"],
        },
        "matrice_confusion": {
            "contrefacons_detectees": cm[0][0],
            "contrefacons_classees_authentiques": cm[0][1],
            "authentiques_rejetes_a_tort": cm[1][0],
            "authentiques_valides": cm[1][1],
        },
        # Comparaison déjà réalisée sur le même jeu de test réel (235 images) —
        # voir MODELES.md pour la méthode complète et les limites. Le chiffre
        # du Random Forest n'est pas recalculé ici, ce modèle n'étant plus en
        # service pour ce score.
        "comparaison_ancien_modele": {
            "random_forest_roc_auc": 0.534,
            "efficientnet_roc_auc": round(info["roc_auc"], 4),
        },
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
