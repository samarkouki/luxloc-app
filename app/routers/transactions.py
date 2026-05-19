"""Router Transactions — Réservation et gestion des locations"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from app.database import get_db
from app.schemas.schemas import TransactionCreate, TransactionResponse
from app.models.models import Article, Transaction, TransactionStatus, Verification, VerificationStatus

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
        "message": "Réservation confirmée. Le paiement sera prélevé à la confirmation du propriétaire."
    }


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
    """Historique des locations d'un utilisateur"""
    if role == "renter":
        txs = db.query(Transaction).filter(Transaction.renter_id == user_id).all()
    else:
        from sqlalchemy import and_
        txs = db.query(Transaction).join(Article).filter(Article.owner_id == user_id).all()

    return [{"id": t.id, "status": t.status.value, "total": t.total_amount, "renter_id": t.renter_id, "article_id": t.article_id, "nb_days": t.nb_days, "created_at": t.created_at} for t in txs]
