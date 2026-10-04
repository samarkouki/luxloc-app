"""Router Transactions — Réservation et gestion des locations"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
from app.database import get_db
from app.schemas.schemas import TransactionCreate, TransactionResponse, TransactionAdminUpdate
from app.models.models import Article, Transaction, TransactionStatus, Verification, VerificationStatus, User
from app.dependencies import require_admin, log_admin_action

router = APIRouter()


@router.post("/", status_code=201)
def create_transaction(tx: TransactionCreate, renter_id: int, db: Session = Depends(get_db)):
    """Créer une réservation (locataire)"""
    article = db.query(Article).filter(Article.id == tx.article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article non trouvé")
    if not article.is_available:
        raise HTTPException(status_code=400, detail="Article non disponible (non vérifié ou déjà loué)")
    if article.owner_id == renter_id:
        raise HTTPException(status_code=400, detail="Vous ne pouvez pas louer votre propre article")

    nb_days = (tx.date_end - tx.date_start).days
    if nb_days < 1:
        raise HTTPException(status_code=400, detail="Durée minimum : 1 jour")

    subtotal = article.price_per_day * nb_days
    insurance = 25.0
    total = subtotal + insurance
    commission = total * 0.20

    transaction = Transaction(
        article_id=article.id,
        renter_id=renter_id,
        status=TransactionStatus.CONFIRMED,
        date_start=tx.date_start,
        date_end=tx.date_end,
        nb_days=nb_days,
        price_per_day=article.price_per_day,
        insurance_fee=insurance,
        total_amount=round(total, 2),
        commission=round(commission, 2),
        guarantee_amount=round(article.price_per_day * 5, 2),  # Caution = 5 jours
    )
    db.add(transaction)
    article.is_available = False  # Plus dispo pendant la location
    db.commit()
    db.refresh(transaction)

    return {
        "transaction_id": transaction.id,
        "status": "confirmed",
        "summary": {
            "article": article.title,
            "brand": article.brand,
            "dates": f"{tx.date_start.strftime('%d/%m/%Y')} → {tx.date_end.strftime('%d/%m/%Y')}",
            "nb_days": nb_days,
            "subtotal": subtotal,
            "insurance": insurance,
            "total": round(total, 2),
            "guarantee_blocked": transaction.guarantee_amount,
        },
        "message": "Réservation enregistrée. Il reste à régler le paiement."
    }


@router.post("/{transaction_id}/pay")
def pay_transaction(transaction_id: int, renter_id: int, db: Session = Depends(get_db)):
    """
    Paiement de DÉMONSTRATION : aucun prestataire de paiement n'est branché,
    aucune carte n'est débitée et le numéro de carte n'est jamais transmis
    au serveur. On enregistre seulement une référence de paiement fictive
    (préfixe « demo_ ») pour que la réservation porte la trace du paiement.
    Le champ stripe_payment_id est réutilisé pour stocker cette référence ;
    il accueillera l'identifiant Stripe réel le jour où le paiement sera branché.
    """
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Réservation introuvable")
    if tx.renter_id != renter_id:
        raise HTTPException(status_code=403, detail="Cette réservation appartient à un autre compte")
    if tx.status not in (TransactionStatus.PENDING, TransactionStatus.CONFIRMED):
        raise HTTPException(status_code=400, detail="Cette réservation ne peut plus être payée")
    if tx.stripe_payment_id:
        return {"transaction_id": tx.id, "reference": tx.stripe_payment_id, "deja_paye": True}

    tx.stripe_payment_id = "demo_" + uuid.uuid4().hex[:12]
    db.commit()
    return {"transaction_id": tx.id, "reference": tx.stripe_payment_id, "deja_paye": False}


@router.get("/{transaction_id}")
def get_transaction(transaction_id: int, db: Session = Depends(get_db)):
    """Détail d'une transaction"""
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction non trouvée")
    article = db.query(Article).filter(Article.id == tx.article_id).first()
    return {
        "id": tx.id,
        "article": {"id": article.id, "title": article.title, "brand": article.brand},
        "status": tx.status.value,
        "dates": {"start": tx.date_start, "end": tx.date_end, "nb_days": tx.nb_days},
        "pricing": {
            "price_per_day": tx.price_per_day,
            "insurance": tx.insurance_fee,
            "total": tx.total_amount,
            "guarantee": tx.guarantee_amount,
        },
        "created_at": tx.created_at,
    }


@router.get("/user/{user_id}")
def user_transactions(user_id: int, role: str = "renter", db: Session = Depends(get_db)):
    """Historique des locations d'un utilisateur, enrichi avec les infos article."""
    from app.models.models import ArticlePhoto
    if role == "renter":
        txs = db.query(Transaction).filter(Transaction.renter_id == user_id).all()
    else:
        txs = db.query(Transaction).join(Article).filter(Article.owner_id == user_id).all()

    result = []
    for t in txs:
        article = db.query(Article).filter(Article.id == t.article_id).first()
        # Photo principale de l'article (overview en base64)
        overview = (db.query(ArticlePhoto)
                    .filter(ArticlePhoto.article_id == t.article_id,
                            ArticlePhoto.photo_type == "overview",
                            ArticlePhoto.image_data.isnot(None))
                    .first())
        # Jours restants avant la fin de la location
        jours_restants = None
        if t.date_end:
            delta = (t.date_end - datetime.utcnow()).days
            jours_restants = max(0, delta)
        result.append({
            "id": t.id,
            "status": t.status.value,
            "total": t.total_amount,
            "renter_id": t.renter_id,
            "article_id": t.article_id,
            "nb_days": t.nb_days,
            "created_at": t.created_at,
            "date_start": t.date_start,
            "date_end": t.date_end,
            "jours_restants": jours_restants,
            # Infos article pour l'affichage
            "article_title": article.title if article else "Article",
            "article_brand": article.brand if article else "",
            "article_photo": overview.image_data if overview else None,
        })
    return result


@router.patch("/{transaction_id}")
def admin_update_transaction(transaction_id: int, payload: TransactionAdminUpdate,
                              admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """
    Change le statut d'une location (administration) : annuler ou clôturer
    typiquement. Si la location est annulée ou clôturée, l'article
    redevient disponible (sauf s'il a été archivé entre-temps).
    """
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction non trouvée")
    try:
        new_status = TransactionStatus(payload.status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Statut invalide")

    old_status = tx.status
    tx.status = new_status
    if new_status in (TransactionStatus.CANCELLED, TransactionStatus.COMPLETED):
        article = db.query(Article).filter(Article.id == tx.article_id).first()
        if article and not article.is_archived:
            article.is_available = True

    log_admin_action(db, admin.id, "update_transaction", "transaction", tx.id,
                      f"{old_status.value} → {new_status.value}")
    db.commit()

    return {"message": "Statut de la location mis à jour", "transaction_id": tx.id, "status": new_status.value}
