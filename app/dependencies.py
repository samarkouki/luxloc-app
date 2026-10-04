"""
Dépendances partagées entre les routers — autorisation admin.

⚠️ SÉCURITÉ — CONTRÔLE VOLONTAIREMENT SIMPLE
Cette API n'a aujourd'hui aucun système de jetons (JWT, session cookie...).
Comme le reste de l'application transmet déjà l'identité de l'appelant en
clair via un paramètre (owner_id, renter_id...), require_admin suit la même
convention : l'appelant passe son propre id (admin_id) et on vérifie en
base qu'il correspond bien à un compte actif de rôle admin.
C'est un choix délibéré pour ce prototype, PAS un mécanisme sûr : quiconque
connaît un admin_id valide peut aujourd'hui l'utiliser. En production, ce
paramètre doit être remplacé par une authentification par jeton vérifiée
côté serveur (JWT signé, session), qui empêcherait toute usurpation.
"""
from fastapi import Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.models import User, UserRole, AdminAction


def require_admin(
    admin_id: int = Query(..., description="Identifiant de l'administrateur appelant"),
    db: Session = Depends(get_db),
) -> User:
    """Vérifie que l'appelant est un compte admin actif, sinon 403."""
    admin = db.query(User).filter(User.id == admin_id).first()
    if not admin or admin.role != UserRole.ADMIN or not admin.is_active:
        raise HTTPException(status_code=403, detail="Accès réservé aux administrateurs")
    return admin


def log_admin_action(db: Session, admin_id: int, action: str, cible_type: str,
                      cible_id: int = None, detail: str = None):
    """
    Enregistre une action d'administration dans le journal (AdminAction).
    N'appelle pas db.commit() — l'appelant commite dans la même transaction
    que le reste de son changement, pour que le journal et l'action restent
    cohérents (tout ou rien).
    """
    db.add(AdminAction(admin_id=admin_id, action=action, cible_type=cible_type,
                        cible_id=cible_id, detail=detail))
