"""
app/admin.py

Espace admin LUXLOC construit avec SQLAdmin, branché directement sur vos
modèles SQLAlchemy (app.models.models). Fournit des vues CRUD pour
Users, Articles, Photos, Verifications, Transactions et Retours.

Protection : seul un compte avec role == ADMIN peut se connecter à
/admin (login par email + mot de passe, réutilise le hash existant du
projet — voir la note de sécurité plus bas).

Activation dans main.py :

    from app.database import engine
    from app.admin import setup_admin
    setup_admin(app, engine)
"""
import os
import hashlib
from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from app.admin_dashboard import DashboardView
from app.database import SessionLocal
from app.models.models import (
    User, UserRole, Article, ArticlePhoto, Verification,
    Transaction, ReturnRecord,
)


def _hash_pw(pw: str) -> str:
    # NOTE SÉCURITÉ : SHA256 nu, cohérent avec le reste du projet pour
    # l'instant (voir app/routers/users.py). Ce n'est pas un hash adapté
    # aux mots de passe (pas de sel, trop rapide à bruteforcer) — à
    # migrer vers passlib[bcrypt] en même temps que le reste de l'auth.
    return hashlib.sha256(pw.encode()).hexdigest()


class AdminAuth(AuthenticationBackend):
    """
    Authentification de l'espace /admin.
    Un seul rôle autorisé à se connecter : UserRole.ADMIN.
    """

    async def login(self, request: Request) -> bool:
        form = await request.form()
        email = form.get("username")
        password = form.get("password")
        if not email or not password:
            return False

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
        finally:
            db.close()

        if not user or user.role != UserRole.ADMIN:
            return False
        if user.hashed_password != _hash_pw(password):
            return False

        request.session.update({"admin_user_id": user.id})
        return True

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        user_id = request.session.get("admin_user_id")
        if not user_id:
            return False

        db = SessionLocal()
        try:
            user = (db.query(User)
                    .filter(User.id == user_id, User.role == UserRole.ADMIN)
                    .first())
        finally:
            db.close()
        return user is not None


# ════════════════════════════════════════
# VUES — UTILISATEURS
# ════════════════════════════════════════
class UserAdmin(ModelView, model=User):
    name = "Utilisateur"
    name_plural = "Utilisateurs"
    icon = "fa-solid fa-user"
    category = "Utilisateurs"

    column_list = [User.id, User.email, User.first_name, User.last_name,
                   User.role, User.is_verified, User.trust_score, User.created_at]
    column_searchable_list = [User.email, User.first_name, User.last_name]
    column_sortable_list = [User.id, User.trust_score, User.created_at]
    column_default_sort = [(User.created_at, True)]

    # On ne veut jamais éditer/afficher le hash du mot de passe depuis l'admin
    form_excluded_columns = [User.hashed_password, User.articles, User.rentals]
    column_details_exclude_list = [User.hashed_password]


# ════════════════════════════════════════
# VUES — CATALOGUE
# ════════════════════════════════════════
class ArticleAdmin(ModelView, model=Article):
    name = "Article"
    name_plural = "Articles"
    icon = "fa-solid fa-bag-shopping"
    category = "Catalogue"

    column_list = [Article.id, Article.title, Article.brand, Article.category,
                   Article.price_per_day, Article.retail_price,
                   Article.is_available, Article.owner, Article.created_at]
    column_searchable_list = [Article.title, Article.brand, Article.model]
    column_sortable_list = [Article.id, Article.price_per_day,
                             Article.retail_price, Article.created_at]
    column_default_sort = [(Article.created_at, True)]

    # Les champs base64 (facture, pièce d'identité) sont énormes et pas
    # exploitables tels quels dans un formulaire HTML — on les exclut.
    form_excluded_columns = [Article.invoice_data, Article.id_document_data,
                              Article.photos, Article.verification, Article.transactions]
    column_details_exclude_list = [Article.invoice_data, Article.id_document_data]


class ArticlePhotoAdmin(ModelView, model=ArticlePhoto):
    name = "Photo"
    name_plural = "Photos"
    icon = "fa-solid fa-image"
    category = "Catalogue"

    column_list = [ArticlePhoto.id, ArticlePhoto.article_id,
                   ArticlePhoto.photo_type, ArticlePhoto.uploaded_at]
    # Le base64 de l'image polluerait complètement la liste/le détail
    column_details_exclude_list = [ArticlePhoto.image_data]
    form_excluded_columns = [ArticlePhoto.image_data]
    can_create = False


class VerificationAdmin(ModelView, model=Verification):
    name = "Vérification"
    name_plural = "Vérifications IA"
    icon = "fa-solid fa-shield-halved"
    category = "Catalogue"

    column_list = [Verification.id, Verification.article_id, Verification.status,
                   Verification.score_global, Verification.pipeline_version,
                   Verification.processing_time_ms, Verification.created_at]
    column_sortable_list = [Verification.score_global, Verification.created_at]
    column_default_sort = [(Verification.created_at, True)]
    can_create = False


# ════════════════════════════════════════
# VUES — VENTES / LOCATIONS
# ════════════════════════════════════════
class TransactionAdmin(ModelView, model=Transaction):
    name = "Transaction"
    name_plural = "Ventes & locations"
    icon = "fa-solid fa-cash-register"
    category = "Ventes"

    column_list = [Transaction.id, Transaction.article, Transaction.renter,
                   Transaction.status, Transaction.date_start, Transaction.date_end,
                   Transaction.total_amount, Transaction.commission, Transaction.created_at]
    column_sortable_list = [Transaction.total_amount, Transaction.commission,
                             Transaction.created_at, Transaction.date_start]
    column_default_sort = [(Transaction.created_at, True)]
    can_create = False  # Les transactions se créent via le flux de réservation, pas à la main


class ReturnRecordAdmin(ModelView, model=ReturnRecord):
    name = "Retour"
    name_plural = "Retours"
    icon = "fa-solid fa-rotate-left"
    category = "Ventes"

    column_list = [ReturnRecord.id, ReturnRecord.transaction_id, ReturnRecord.condition,
                   ReturnRecord.guarantee_released, ReturnRecord.guarantee_deducted,
                   ReturnRecord.created_at]
    column_default_sort = [(ReturnRecord.created_at, True)]
    can_create = False


def setup_admin(app, engine):
    secret_key = os.getenv("ADMIN_SECRET_KEY", "change-me-in-production")
    auth_backend = AdminAuth(secret_key=secret_key)

    admin = Admin(app, engine, authentication_backend=auth_backend,
                  title="LUXLOC — Espace Admin")

    admin.add_view(DashboardView)   
    admin.add_view(UserAdmin)
    admin.add_view(ArticleAdmin)
    admin.add_view(ArticlePhotoAdmin)
    admin.add_view(VerificationAdmin)
    admin.add_view(TransactionAdmin)
    admin.add_view(ReturnRecordAdmin)

    return admin