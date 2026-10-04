"""Router Utilisateurs"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.database import get_db
from app.schemas.schemas import (UserCreate, UserResponse, UserLogin, UserAdminUpdate,
                                 PasswordForgotRequest, PasswordResetRequest)
from app.models.models import User, Article, Transaction, TransactionStatus, PasswordResetToken
from app.dependencies import require_admin, log_admin_action
from app.services import email as mailer
import hashlib
import os
import secrets

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


RESET_VALIDITY = timedelta(minutes=30)
RESET_COOLDOWN = timedelta(seconds=60)


def _reset_base_url(request: Request):
    """
    Adresse du site à mettre dans le lien. APP_BASE_URL prime si elle est
    définie ; sinon on reprend l'adresse de la requête, en forçant https hors
    local (derrière le proxy Railway, uvicorn voit la requête en http).
    """
    base = os.getenv("APP_BASE_URL")
    if base:
        return base.rstrip("/")
    base = str(request.base_url).rstrip("/")
    if base.startswith("http://") and "localhost" not in base and "127.0.0.1" not in base:
        base = "https://" + base[len("http://"):]
    return base


@router.post("/password/forgot")
def password_forgot(payload: PasswordForgotRequest, request: Request, db: Session = Depends(get_db)):
    """
    Envoie un lien de réinitialisation par e-mail. La réponse est toujours la
    même, que l'adresse existe ou non, pour ne pas révéler quels comptes
    existent. Un nouveau lien invalide les précédents ; au plus un envoi par
    minute et par compte.
    """
    message = {"message": "Si un compte existe avec cette adresse, un e-mail de réinitialisation vient d'être envoyé."}
    user = db.query(User).filter(User.email == payload.email.strip().lower()).first() \
        or db.query(User).filter(User.email == payload.email.strip()).first()
    if not user or not user.is_active:
        return message

    now = datetime.utcnow()
    recent = (db.query(PasswordResetToken)
              .filter(PasswordResetToken.user_id == user.id,
                      PasswordResetToken.created_at > now - RESET_COOLDOWN)
              .first())
    if recent:
        return message

    db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user.id,
        PasswordResetToken.used_at.is_(None),
    ).update({PasswordResetToken.used_at: now}, synchronize_session=False)

    token = secrets.token_urlsafe(32)
    db.add(PasswordResetToken(
        user_id=user.id,
        token_hash=hashlib.sha256(token.encode()).hexdigest(),
        expires_at=now + RESET_VALIDITY,
    ))
    db.commit()

    link = f"{_reset_base_url(request)}/?reset={token}"
    if not mailer.is_configured():
        # Sans service d'e-mail : en local on affiche le lien pour pouvoir
        # tester ; en production on ne l'écrit jamais dans les journaux.
        if db.bind.dialect.name == "sqlite":
            print(f"[email] non configuré — lien de réinitialisation pour {user.email} : {link}")
        else:
            print("[email] non configuré — RESEND_API_KEY absente, aucun e-mail envoyé")
        return message

    html = (
        f"<p>Bonjour {user.first_name},</p>"
        "<p>Vous avez demandé à réinitialiser votre mot de passe LUXLOC.</p>"
        f'<p><a href="{link}">Choisir un nouveau mot de passe</a></p>'
        "<p>Ce lien est valable 30 minutes et ne fonctionne qu'une fois. "
        "Si vous n'êtes pas à l'origine de cette demande, ignorez ce message : "
        "votre mot de passe actuel reste inchangé.</p>"
        "<p>L'équipe LUXLOC</p>"
    )
    try:
        mailer.send_email(user.email, "Réinitialisation de votre mot de passe LUXLOC", html)
    except Exception as e:
        print(f"[email] échec de l'envoi : {type(e).__name__}: {e}")
    return message


@router.post("/password/reset")
def password_reset(payload: PasswordResetRequest, db: Session = Depends(get_db)):
    """Définit un nouveau mot de passe à partir d'un lien reçu par e-mail."""
    invalid = HTTPException(status_code=400,
                            detail="Ce lien n'est plus valable. Demandez-en un nouveau depuis l'écran de connexion.")
    row = (db.query(PasswordResetToken)
           .filter(PasswordResetToken.token_hash == hashlib.sha256(payload.token.encode()).hexdigest())
           .first())
    now = datetime.utcnow()
    if not row or row.used_at is not None or row.expires_at < now:
        raise invalid
    user = db.query(User).filter(User.id == row.user_id).first()
    if not user or not user.is_active:
        raise invalid

    user.hashed_password = hash_pw(payload.new_password)
    row.used_at = now
    db.commit()
    return {"message": "Mot de passe modifié. Vous pouvez vous connecter."}


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
