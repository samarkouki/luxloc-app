"""Router Retours — Vérification de l'état au retour"""
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.schemas import ReturnCreate, ReturnResponse
from app.models.models import Transaction, TransactionStatus, ReturnRecord, ReturnCondition, Article, User

router = APIRouter()

CONDITION_SCALE = ["Neuf", "Excellent", "Très bon", "Bon", "Correct", "Endommagé"]


def _degrade_condition(cond: str, steps: int = 1) -> str:
    idx = CONDITION_SCALE.index(cond) if cond in CONDITION_SCALE else 1
    return CONDITION_SCALE[min(idx + steps, len(CONDITION_SCALE) - 1)]


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

    # Ponctualité + traçabilité de l'état de l'article
    article = db.query(Article).filter(Article.id == tx.article_id).first()
    condition_at_rental = (article.condition if article else None) or "Excellent"
    if condition == ReturnCondition.DAMAGED:
        condition_at_return = "Endommagé"
    elif condition == ReturnCondition.PENDING_REVIEW:
        condition_at_return = _degrade_condition(condition_at_rental, 1)
    else:
        condition_at_return = condition_at_rental

    actual_return_date = datetime.utcnow()
    returned_on_time = actual_return_date <= tx.date_end

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
        condition_at_rental=condition_at_rental,
        condition_at_return=condition_at_return,
        actual_return_date=actual_return_date,
        returned_on_time=returned_on_time,
    )
    db.add(record)

    # Mettre à jour le statut de la transaction
    tx.status = TransactionStatus.COMPLETED

    # Remettre l'article en disponibilité
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


@router.get("/platform/stats")
def returns_stats(months_back: int = 12, db: Session = Depends(get_db)):
    """
    Statistiques de ponctualité et d'évolution de l'état des articles au retour.
    Alimente le graphique 'Ponctualité des retours' du dashboard admin.
    """
    since = datetime.utcnow() - timedelta(days=months_back * 30)
    rows = (
        db.query(ReturnRecord, Transaction)
        .join(Transaction, ReturnRecord.transaction_id == Transaction.id)
        .filter(Transaction.date_end >= since)
        .all()
    )

    total = len(rows)
    on_time = sum(1 for r, _ in rows if r.returned_on_time)
    late = total - on_time

    monthly = {}
    for r, tx in rows:
        key = tx.date_end.strftime("%Y-%m")
        bucket = monthly.setdefault(key, {"a_temps": 0, "en_retard": 0})
        bucket["a_temps" if r.returned_on_time else "en_retard"] += 1
    months_sorted = sorted(monthly.keys())

    evolution = {"ameliore": 0, "identique": 0, "degrade": 0}
    for r, _ in rows:
        if not r.condition_at_rental or not r.condition_at_return:
            continue
        i1 = CONDITION_SCALE.index(r.condition_at_rental) if r.condition_at_rental in CONDITION_SCALE else 1
        i2 = CONDITION_SCALE.index(r.condition_at_return) if r.condition_at_return in CONDITION_SCALE else 1
        if i2 > i1:
            evolution["degrade"] += 1
        elif i2 < i1:
            evolution["ameliore"] += 1
        else:
            evolution["identique"] += 1

    return {
        "total_retours": total,
        "a_temps": on_time,
        "en_retard": late,
        "taux_ponctualite": round(on_time / total * 100, 1) if total else 0,
        "mensuel": {
            "labels": months_sorted,
            "a_temps": [monthly[m]["a_temps"] for m in months_sorted],
            "en_retard": [monthly[m]["en_retard"] for m in months_sorted],
        },
        "evolution_etat": evolution,
    }


@router.get("/platform/detail")
def returns_detail(limit: int = 30, db: Session = Depends(get_db)):
    """
    Table détaillée des retours : article, client, dates de location/retour
    (prévue et réelle), ponctualité, état à la location et au retour.
    """
    rows = (
        db.query(ReturnRecord, Transaction, Article, User)
        .join(Transaction, ReturnRecord.transaction_id == Transaction.id)
        .join(Article, Transaction.article_id == Article.id)
        .join(User, Transaction.renter_id == User.id)
        .order_by(ReturnRecord.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "article": a.title,
            "marque": a.brand,
            "client": f"{u.first_name} {u.last_name}",
            "date_location": tx.date_start,
            "duree_jours": tx.nb_days,
            "retour_prevu": tx.date_end,
            "retour_reel": r.actual_return_date,
            "a_temps": r.returned_on_time,
            "etat_location": r.condition_at_rental,
            "etat_retour": r.condition_at_return,
        }
        for r, tx, a, u in rows
    ]
