"""Router Retours — Vérification de l'état au retour"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.schemas import ReturnCreate, ReturnResponse
from app.models.models import Transaction, TransactionStatus, ReturnRecord, ReturnCondition, Article

router = APIRouter()


@router.post("/", status_code=201)
def create_return(ret: ReturnCreate, db: Session = Depends(get_db)):
    """Enregistrer le retour d'un article avec checklist"""
    tx = db.query(Transaction).filter(Transaction.id == ret.transaction_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction non trouvée")
    if tx.status not in [TransactionStatus.ACTIVE, TransactionStatus.CONFIRMED]:
        raise HTTPException(status_code=400, detail="Cette location n'est pas en cours")

    # Évaluer la condition
    all_ok = all([ret.general_condition, ret.cleanliness, ret.hardware_intact, ret.accessories_returned])
    if all_ok:
        condition = ReturnCondition.CONFORM
    elif sum([ret.general_condition, ret.cleanliness, ret.hardware_intact, ret.accessories_returned]) >= 3:
        condition = ReturnCondition.PENDING_REVIEW
    else:
        condition = ReturnCondition.DAMAGED

    # Calculer la retenue sur garantie
    guarantee_deducted = 0.0
    if condition == ReturnCondition.DAMAGED:
        guarantee_deducted = tx.guarantee_amount * 0.5  # 50% retenu
    elif condition == ReturnCondition.PENDING_REVIEW:
        guarantee_deducted = 0.0  # En attente de décision expert

    record = ReturnRecord(
        transaction_id=tx.id,
        general_condition=ret.general_condition,
        cleanliness=ret.cleanliness,
        hardware_intact=ret.hardware_intact,
        accessories_returned=ret.accessories_returned,
        condition=condition,
        notes=ret.notes,
        guarantee_released=(condition == ReturnCondition.CONFORM),
        guarantee_deducted=round(guarantee_deducted, 2),
    )
    db.add(record)

    # Mettre à jour le statut de la transaction
    tx.status = TransactionStatus.COMPLETED

    # Remettre l'article en disponibilité
    article = db.query(Article).filter(Article.id == tx.article_id).first()
    if article:
        article.is_available = True

    db.commit()
    db.refresh(record)

    verdict = {
        ReturnCondition.CONFORM: "Retour conforme — Garantie intégralement restituée",
        ReturnCondition.PENDING_REVIEW: "Retour avec réserves — Un expert va examiner le dossier",
        ReturnCondition.DAMAGED: "Dommages constatés — Retenue partielle sur la garantie",
    }

    return {
        "return_id": record.id,
        "condition": condition.value,
        "verdict": verdict[condition],
        "checklist": {
            "general_condition": ret.general_condition,
            "cleanliness": ret.cleanliness,
            "hardware_intact": ret.hardware_intact,
            "accessories_returned": ret.accessories_returned,
        },
        "guarantee": {
            "total": tx.guarantee_amount,
            "released": record.guarantee_released,
            "deducted": record.guarantee_deducted,
            "refunded": round(tx.guarantee_amount - record.guarantee_deducted, 2),
        }
    }


@router.get("/{transaction_id}")
def get_return(transaction_id: int, db: Session = Depends(get_db)):
    """Récupérer le détail d'un retour"""
    record = db.query(ReturnRecord).filter(ReturnRecord.transaction_id == transaction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Retour non trouvé")
    return record
