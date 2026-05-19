"""Router Utilisateurs"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.schemas import UserCreate, UserResponse, UserLogin
from app.models.models import User
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
