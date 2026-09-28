"""Router Articles — CRUD + Filtrage catalogue"""
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Body
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
from app.database import get_db
from app.schemas.schemas import ArticleCreate, ArticleResponse, ArticleUpdate, CategoryEnum
from app.models.models import Article, ArticlePhoto, Verification, VerificationStatus, User
from app.dependencies import require_admin, log_admin_action

router = APIRouter()


@router.post("/", response_model=ArticleResponse, status_code=201)
def create_article(article: ArticleCreate, owner_id: int, db: Session = Depends(get_db)):
    """Créer un nouvel article (propriétaire)"""
    db_article = Article(
        owner_id=owner_id,
        title=article.title,
        brand=article.brand,
        model=article.model,
        category=article.category,
        description=article.description,
        price_per_day=article.price_per_day,
        retail_price=article.retail_price,
        condition=article.condition,
        is_available=False,  # Pas dispo tant que pas vérifié
    )
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return _enrich_article(db_article, db)


@router.get("/", response_model=List[ArticleResponse])
def list_articles(
    category: Optional[CategoryEnum] = None,
    brand: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    verified_only: bool = True,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """Catalogue avec filtres (locataire)"""
    # Un article archivé par un admin ne doit plus jamais apparaître ici,
    # quels que soient les autres filtres.
    query = db.query(Article).filter(Article.is_archived == False)

    if verified_only:
        query = query.filter(Article.is_available == True)
    if category:
        query = query.filter(Article.category == category)
    if brand:
        query = query.filter(Article.brand.ilike(f"%{brand}%"))
    if min_price:
        query = query.filter(Article.price_per_day >= min_price)
    if max_price:
        query = query.filter(Article.price_per_day <= max_price)

    articles = query.offset(skip).limit(limit).all()
    return [_enrich_article(a, db) for a in articles]


@router.get("/categories/stats")
def category_stats(db: Session = Depends(get_db)):
    """Nombre d'articles par catégorie"""
    from sqlalchemy import func
    stats = db.query(
        Article.category, func.count(Article.id)
    ).filter(Article.is_available == True).group_by(Article.category).all()
    return {cat: count for cat, count in stats}


@router.get("/{article_id}", response_model=ArticleResponse)
def get_article(article_id: int, db: Session = Depends(get_db)):
    """Fiche détaillée d'un article"""
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")
    return _enrich_article(article, db)


@router.post("/{article_id}/photos")
def upload_photos(
    article_id: int,
    photo_type: str = Query(..., description="overview, logo, label, stitching"),
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
):
    """
    Upload d'une photo pour un article.
    Le corps de la requête peut contenir {"image_data": "<base64 ou data URI>"}.
    L'image est conservée pour l'analyse réelle du pipeline de vérification IA.
    En production, l'image serait stockée sur Google Cloud Storage.
    """
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")

    image_data = payload.get("image_data") if isinstance(payload, dict) else None

    photo = ArticlePhoto(
        article_id=article_id,
        photo_type=photo_type,
        file_path=f"gs://luxloc-photos/{article_id}/{photo_type}.jpg",
        image_data=image_data,
    )
    db.add(photo)
    db.commit()
    return {
        "message": f"Photo '{photo_type}' uploadée",
        "photo_id": photo.id,
        "path": photo.file_path,
        "has_image": image_data is not None,
    }


@router.post("/{article_id}/documents")
def upload_documents(
    article_id: int,
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
):
    """
    Upload des documents de confiance du propriétaire :
    - invoice_data       : facture d'achat (base64) — preuve d'achat
    - id_document_data   : pièce d'identité (base64)

    La facture sera analysée par le pipeline de vérification (ELA,
    validation de format, cohérence article). En production les documents
    seraient chiffrés et stockés sur Google Cloud Storage.
    """
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")

    invoice = payload.get("invoice_data") if isinstance(payload, dict) else None
    id_doc = payload.get("id_document_data") if isinstance(payload, dict) else None

    if invoice:
        article.invoice_data = invoice
    if id_doc:
        article.id_document_data = id_doc
    db.commit()

    return {
        "message": "Documents enregistrés",
        "article_id": article_id,
        "has_invoice": article.invoice_data is not None,
        "has_id_document": article.id_document_data is not None,
    }


@router.get("/{article_id}/estimate")
def price_estimate(article_id: int, nb_days: int = Query(ge=1), db: Session = Depends(get_db)):
    """Estimation du prix pour une durée donnée"""
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")

    subtotal = article.price_per_day * nb_days
    insurance = 25.0
    total = subtotal + insurance
    commission = total * 0.20

    return {
        "nb_days": nb_days,
        "price_per_day": article.price_per_day,
        "subtotal": subtotal,
        "insurance_fee": insurance,
        "total": total,
        "commission": round(commission, 2),
        "owner_receives": round(total - commission, 2),
    }

@router.get("/platform/stats")
def platform_stats(db: Session = Depends(get_db)):
    """
    Statistiques globales de la plateforme LUXLOC.
    Alimente la section Stats de l'app et, à terme, le dashboard BI externe.
    """
    from sqlalchemy import func
    from app.models.models import Transaction, User

    # Les articles supprimés (archivés) sortent de toutes les statistiques.
    actifs = db.query(Article.id).filter(Article.is_archived == False)
    total_articles = actifs.count()
    total_users = db.query(User).filter(User.is_active == True).count()
    total_transactions = db.query(Transaction).count()

    # Chaque relance d'analyse ajoute une ligne de vérification : on ne
    # compte que la plus récente de chaque article, sinon les relances
    # gonflent les totaux.
    derniere = (db.query(func.max(Verification.id))
                .filter(Verification.article_id.in_(actifs))
                .group_by(Verification.article_id))
    latest = db.query(Verification).filter(Verification.id.in_(derniere)).all()

    # reviewed_by est renseigné quand un expert tranche ; une validation
    # sans reviewed_by vient donc directement du pipeline IA.
    auto = sum(1 for v in latest if v.status == VerificationStatus.VERIFIED and not v.reviewed_by)
    expert = sum(1 for v in latest if v.status == VerificationStatus.VERIFIED and v.reviewed_by)
    review = sum(1 for v in latest if v.status == VerificationStatus.REVIEW)
    rejected = sum(1 for v in latest if v.status == VerificationStatus.REJECTED)
    verified = auto + expert

    total_verifs = len(latest)
    taux_auto = round(auto / total_verifs * 100, 1) if total_verifs else 0

    scores = [v.score_global for v in latest if v.score_global is not None]
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0

    cat_rows = (db.query(Article.category, func.count(Article.id))
                .filter(Article.is_archived == False)
                .group_by(Article.category).all())
    par_categorie = {
        (c.value if hasattr(c, "value") else c): n for c, n in cat_rows
    }

    brand_rows = (db.query(Article.brand, func.count(Article.id))
                  .filter(Article.is_archived == False)
                  .group_by(Article.brand)
                  .order_by(func.count(Article.id).desc()).all())
    par_marque = {b: n for b, n in brand_rows}

    # Valeur neuve du catalogue (somme des prix de vente d'origine)
    valeur_catalogue = (db.query(func.sum(Article.retail_price))
                        .filter(Article.is_archived == False).scalar() or 0)

    # Revenus de commission (20% du total des transactions)
    revenus_commission = db.query(func.sum(Transaction.commission)).scalar() or 0

    return {
        "total_articles": total_articles,
        "total_utilisateurs": total_users,
        "total_locations": total_transactions,
        "verification": {
            "verifies": verified,
            "verifies_auto": auto,
            "verifies_expert": expert,
            "en_examen": review,
            "rejetes": rejected,
            "sans_analyse": total_articles - total_verifs,
            "taux_auto_validation": taux_auto,
        },
        "score_ia_moyen": avg_score,
        "par_categorie": par_categorie,
        "par_marque": par_marque,
        "valeur_catalogue": round(valeur_catalogue, 0),
        "revenus_commission": round(revenus_commission, 2),
    }


@router.patch("/{article_id}", response_model=ArticleResponse)
def update_article(article_id: int, payload: ArticleUpdate,
                    admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Modifier un article (administration) : titre, marque, modèle, catégorie, prix, état, disponibilité."""
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")

    changes = payload.dict(exclude_unset=True)
    for field, value in changes.items():
        setattr(article, field, value)

    if changes:
        log_admin_action(db, admin.id, "update_article", "article", article.id,
                          ", ".join(f"{k}={v}" for k, v in changes.items()))
    db.commit()
    db.refresh(article)
    return _enrich_article(article, db)


@router.delete("/{article_id}")
def archive_article(article_id: int, admin: User = Depends(require_admin),
                     db: Session = Depends(get_db)):
    """
    Archive un article (administration). Aucune suppression physique :
    l'article disparaît du catalogue public mais reste restaurable.
    """
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")

    article.is_archived = True
    article.archived_at = datetime.utcnow()
    article.is_available = False
    log_admin_action(db, admin.id, "archive_article", "article", article.id, article.title)
    db.commit()

    return {"message": "Article archivé", "article_id": article.id, "is_archived": True}


@router.post("/{article_id}/restore")
def restore_article(article_id: int, admin: User = Depends(require_admin),
                     db: Session = Depends(get_db)):
    """Désarchive un article (administration) — redevient gérable par son propriétaire."""
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")

    article.is_archived = False
    article.archived_at = None
    log_admin_action(db, admin.id, "restore_article", "article", article.id, article.title)
    db.commit()

    return {"message": "Article restauré", "article_id": article.id, "is_archived": False}


def _enrich_article(article: Article, db: Session) -> dict:
    """Ajoute les infos de vérification et la photo principale à un article"""
    verif = db.query(Verification).filter(Verification.article_id == article.id).first()
    # Photo principale : on récupère la photo 'overview' si elle a une image
    overview = (db.query(ArticlePhoto)
                .filter(ArticlePhoto.article_id == article.id,
                        ArticlePhoto.photo_type == "overview",
                        ArticlePhoto.image_data.isnot(None))
                .first())
    return {
        "id": article.id,
        "owner_id": article.owner_id,
        "title": article.title,
        "brand": article.brand,
        "model": article.model,
        "category": article.category.value if hasattr(article.category, 'value') else article.category,
        "description": article.description,
        "price_per_day": article.price_per_day,
        "condition": article.condition,
        "is_available": article.is_available,
        "verification_status": verif.status.value if verif else None,
        "verification_score": verif.score_global if verif else None,
        "photo": overview.image_data if overview else None,
        "created_at": article.created_at,
    }
