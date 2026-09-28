"""Router Utilisateurs"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.schemas import UserCreate, UserResponse, UserLogin, UserAdminUpdate
from app.models.models import User, Article, Transaction, TransactionStatus
from app.dependencies import require_admin, log_admin_action
import hashlib

router = APIRouter()

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_pw(password, hashed):
    return hash_pw(password) == hashed


@router.post("/register", response_model=UserResponse, status_code=201)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """Inscription d'un nouvel utilisateur"""
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email déjà utilisé")
    db_user = User(
        email=user.email,
        hashed_password=hash_pw(user.password),
        first_name=user.first_name,
        last_name=user.last_name,
        phone=user.phone,
        role=user.role,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.post("/login")
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Connexion utilisateur"""
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not verify_pw(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Identifiants incorrects")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Ce compte a été suspendu")
    return {"message": "Connexion réussie", "user_id": user.id, "role": user.role}


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Récupérer un profil utilisateur"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    return user


@router.get("/{user_id}/dashboard")
def get_dashboard(user_id: int, db: Session = Depends(get_db)):
    """Dashboard propriétaire : articles, revenus, locations actives"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    from app.models.models import Article, Transaction, TransactionStatus
    articles = db.query(Article).filter(Article.owner_id == user_id).all()
    active_rentals = db.query(Transaction).filter(
        Transaction.status == TransactionStatus.ACTIVE
    ).count()

    return {
        "user": {"id": user.id, "name": f"{user.first_name} {user.last_name}", "trust_score": user.trust_score},
        "stats": {
            "total_articles": len(articles),
            "available_articles": sum(1 for a in articles if a.is_available),
            "active_rentals": active_rentals,
            "total_revenue": 0,  # Calculé depuis les transactions
        }
    }


@router.patch("/{user_id}", response_model=UserResponse)
def admin_update_user(user_id: int, payload: UserAdminUpdate,
                       admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """
    Modifie le rôle et/ou le statut actif d'un compte (administration).
    Un admin ne peut jamais modifier son propre rôle ni se désactiver
    lui-même — sans quoi il pourrait se retrouver bloqué hors de la
    console qu'il est le seul à pouvoir utiliser.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    if user.id == admin.id:
        if payload.role is not None and payload.role != admin.role.value:
            raise HTTPException(status_code=403,
                                detail="Un administrateur ne peut pas modifier son propre rôle")
        if payload.is_active is False:
            raise HTTPException(status_code=403,
                                detail="Un administrateur ne peut pas se désactiver lui-même")

    changes = []
    if payload.role is not None and payload.role != user.role.value:
        changes.append(f"role : {user.role.value} → {payload.role}")
        user.role = payload.role
    if payload.is_active is not None and payload.is_active != user.is_active:
        changes.append(f"is_active : {user.is_active} → {payload.is_active}")
        user.is_active = payload.is_active

    if changes:
        log_admin_action(db, admin.id, "update_user", "user", user.id, " ; ".join(changes))
    db.commit()
    db.refresh(user)
    return user


@router.delete("/{user_id}")
def deactivate_user(user_id: int, admin: User = Depends(require_admin),
                     db: Session = Depends(get_db)):
    """
    Suspend un compte (administration). Aucune suppression physique —
    le compte est désactivé (is_active=False) et ne peut plus se
    connecter. Refusé si l'utilisateur a une location en cours, en tant
    que locataire ou que propriétaire de l'article loué.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    if user.id == admin.id:
        raise HTTPException(status_code=403,
                            detail="Un administrateur ne peut pas se désactiver lui-même")

    en_cours = [TransactionStatus.ACTIVE, TransactionStatus.CONFIRMED]
    as_renter = db.query(Transaction).filter(
        Transaction.renter_id == user_id, Transaction.status.in_(en_cours)).count()
    as_owner = db.query(Transaction).join(Article).filter(
        Article.owner_id == user_id, Transaction.status.in_(en_cours)).count()
    if as_renter or as_owner:
        raise HTTPException(status_code=400,
                            detail="Impossible : cet utilisateur a une location en cours")

    user.is_active = False
    log_admin_action(db, admin.id, "suspend_user", "user", user.id,
                      f"{user.first_name} {user.last_name}")
    db.commit()

    return {"message": "Utilisateur suspendu", "user_id": user.id, "is_active": False}
