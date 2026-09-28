"""
Router Administration — Consultation pour la console admin.

Toutes les routes de ce fichier sont des lectures (GET) réservées au rôle
admin via app.dependencies.require_admin. Les actions qui modifient l'état
(archiver, suspendre, changer un statut...) vivent dans les routers
existants (articles, users, transactions, verification) au plus près des
modèles qu'elles touchent ; ce fichier ne fait qu'agréger des vues pour la
console.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from app.database import get_db
from app.dependencies import require_admin
from app.models.models import (
    Article, ArticlePhoto, Verification, VerificationStatus,
    Transaction, User, AdminAction,
)

router = APIRouter()


@router.get("/moderation")
def moderation_queue(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """
    File des articles en attente d'examen (dernière vérification en
    statut 'review'), du plus ancien au plus récent. Pour chacun : photo,
    score global, détail des sous-scores et le composant le plus faible.
    """
    verifs = (db.query(Verification)
              .join(Article, Verification.article_id == Article.id)
              .filter(Verification.status == VerificationStatus.REVIEW,
                      Article.is_archived == False)
              .order_by(Verification.created_at.asc())
              .all())

    result = []
    for v in verifs:
        article = db.query(Article).filter(Article.id == v.article_id).first()
        if not article:
            continue
        owner = db.query(User).filter(User.id == article.owner_id).first()
        overview = (db.query(ArticlePhoto)
                    .filter(ArticlePhoto.article_id == article.id,
                            ArticlePhoto.photo_type == "overview",
                            ArticlePhoto.image_data.isnot(None))
                    .first())
        sous_scores = {
            "logo": v.score_logo,
            "coutures": v.score_stitching,
            "texture": v.score_texture,
            "etiquette": v.score_label,
            "regles": v.score_rules,
            "facture": v.score_invoice,
        }
        composant_faible = min(sous_scores, key=lambda k: sous_scores[k] if sous_scores[k] is not None else 1.0)

        result.append({
            "article_id": article.id,
            "verification_id": v.id,
            "title": article.title,
            "brand": article.brand,
            "model": article.model,
            "category": article.category.value if hasattr(article.category, "value") else article.category,
            "price_per_day": article.price_per_day,
            "photo": overview.image_data if overview else None,
            "owner": {"id": owner.id, "name": f"{owner.first_name} {owner.last_name}"} if owner else None,
            "score_global": v.score_global,
            "sous_scores": sous_scores,
            "composant_le_plus_faible": composant_faible,
            "soumis_le": v.created_at,
        })
    return result


@router.get("/articles")
def admin_list_articles(
    statut: Optional[str] = Query(None, description="verified, review ou rejected"),
    category: Optional[str] = None,
    owner_id: Optional[int] = None,
    archived: Optional[bool] = Query(None, description="True: seulement archivés, False: seulement actifs, absent: tous"),
    admin: User = Depends(require_admin), db: Session = Depends(get_db),
):
    """Tous les articles (archivés inclus par défaut), filtrables par statut, catégorie, propriétaire."""
    query = db.query(Article)
    if owner_id is not None:
        query = query.filter(Article.owner_id == owner_id)
    if category:
        query = query.filter(Article.category == category)
    if archived is not None:
        query = query.filter(Article.is_archived == archived)

    result = []
    for a in query.order_by(Article.id.desc()).all():
        verif = (db.query(Verification).filter(Verification.article_id == a.id)
                 .order_by(Verification.id.desc()).first())
        if statut and (not verif or verif.status.value != statut):
            continue
        owner = db.query(User).filter(User.id == a.owner_id).first()
        overview = (db.query(ArticlePhoto)
                    .filter(ArticlePhoto.article_id == a.id,
                            ArticlePhoto.photo_type == "overview",
                            ArticlePhoto.image_data.isnot(None))
                    .first())
        result.append({
            "id": a.id,
            "title": a.title,
            "brand": a.brand,
            "model": a.model,
            "category": a.category.value if hasattr(a.category, "value") else a.category,
            "price_per_day": a.price_per_day,
            "condition": a.condition,
            "is_available": a.is_available,
            "is_archived": a.is_archived,
            "archived_at": a.archived_at,
            "verification_status": verif.status.value if verif else None,
            "verification_score": verif.score_global if verif else None,
            "owner": {"id": owner.id, "name": f"{owner.first_name} {owner.last_name}"} if owner else None,
            "photo": overview.image_data if overview else None,
            "created_at": a.created_at,
        })
    return result


@router.get("/users")
def admin_list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Tous les comptes, avec nombre d'articles possédés et de locations effectuées."""
    result = []
    for u in db.query(User).order_by(User.id.asc()).all():
        nb_articles = db.query(Article).filter(Article.owner_id == u.id).count()
        nb_locations = db.query(Transaction).filter(Transaction.renter_id == u.id).count()
        result.append({
            "id": u.id,
            "name": f"{u.first_name} {u.last_name}",
            "email": u.email,
            "role": u.role.value if hasattr(u.role, "value") else u.role,
            "trust_score": u.trust_score,
            "is_active": u.is_active,
            "nb_articles": nb_articles,
            "nb_locations": nb_locations,
            "created_at": u.created_at,
        })
    return result


@router.get("/transactions")
def admin_list_transactions(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Toutes les locations, avec propriétaire, locataire, article et montant."""
    result = []
    for t in db.query(Transaction).order_by(Transaction.id.desc()).all():
        article = db.query(Article).filter(Article.id == t.article_id).first()
        renter = db.query(User).filter(User.id == t.renter_id).first()
        owner = db.query(User).filter(User.id == article.owner_id).first() if article else None
        result.append({
            "id": t.id,
            "article": {"id": article.id, "title": article.title} if article else None,
            "owner": {"id": owner.id, "name": f"{owner.first_name} {owner.last_name}"} if owner else None,
            "renter": {"id": renter.id, "name": f"{renter.first_name} {renter.last_name}"} if renter else None,
            "status": t.status.value,
            "total_amount": t.total_amount,
            "date_start": t.date_start,
            "date_end": t.date_end,
            "created_at": t.created_at,
        })
    return result


@router.get("/audit")
def admin_audit_log(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Journal des actions d'administration, la plus récente en premier."""
    result = []
    for a in db.query(AdminAction).order_by(AdminAction.id.desc()).limit(200).all():
        who = db.query(User).filter(User.id == a.admin_id).first()
        result.append({
            "id": a.id,
            "admin": f"{who.first_name} {who.last_name}" if who else f"#{a.admin_id}",
            "action": a.action,
            "cible_type": a.cible_type,
            "cible_id": a.cible_id,
            "detail": a.detail,
            "created_at": a.created_at,
        })
    return result
