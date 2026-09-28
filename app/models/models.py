"""Modèles SQLAlchemy — Base de données LUXLOC"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.database import Base


class UserRole(str, enum.Enum):
    OWNER = "owner"
    RENTER = "renter"
    BOTH = "both"
    ADMIN = "admin"


class ArticleCategory(str, enum.Enum):
    BAGS = "bags"
    WATCHES = "watches"
    JEWELRY = "jewelry"
    CLOTHING = "clothing"
    EYEWEAR = "eyewear"


class VerificationStatus(str, enum.Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REVIEW = "review"         # Envoyé à un expert humain
    REJECTED = "rejected"


class TransactionStatus(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    ACTIVE = "active"         # Location en cours
    RETURN_PENDING = "return_pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReturnCondition(str, enum.Enum):
    CONFORM = "conform"
    DAMAGED = "damaged"
    PENDING_REVIEW = "pending_review"


# ════════════════════════════════════
# UTILISATEURS
# ════════════════════════════════════
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    phone = Column(String)
    role = Column(Enum(UserRole), default=UserRole.BOTH)
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)  # False = compte suspendu par un admin
    trust_score = Column(Float, default=0.5)  # Score de confiance 0-1
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relations
    articles = relationship("Article", back_populates="owner")
    rentals = relationship("Transaction", back_populates="renter", foreign_keys="Transaction.renter_id")


# ════════════════════════════════════
# ARTICLES
# ════════════════════════════════════
class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    brand = Column(String, nullable=False, index=True)
    model = Column(String)
    category = Column(Enum(ArticleCategory), nullable=False, index=True)
    description = Column(Text)
    price_per_day = Column(Float, nullable=False)
    retail_price = Column(Float)  # Prix neuf pour cohérence
    condition = Column(String, default="Excellent")
    is_available = Column(Boolean, default=False)  # Disponible après vérification
    is_archived = Column(Boolean, default=False)   # Archivé par un admin — masqué du catalogue
    archived_at = Column(DateTime, nullable=True)
    invoice_data = Column(Text)        # Facture d'achat (base64) — preuve d'achat
    id_document_data = Column(Text)    # Pièce d'identité du propriétaire (base64)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relations
    owner = relationship("User", back_populates="articles")
    photos = relationship("ArticlePhoto", back_populates="article")
    verification = relationship("Verification", back_populates="article", uselist=False)
    transactions = relationship("Transaction", back_populates="article")


class ArticlePhoto(Base):
    __tablename__ = "article_photos"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    photo_type = Column(String, nullable=False)  # overview, logo, label, stitching
    file_path = Column(String, nullable=False)    # Chemin GCS
    image_data = Column(Text, nullable=True)       # Image base64 (PoC — en prod : GCS)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    article = relationship("Article", back_populates="photos")


# ════════════════════════════════════
# VÉRIFICATION IA
# ════════════════════════════════════
class Verification(Base):
    __tablename__ = "verifications"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    status = Column(Enum(VerificationStatus), default=VerificationStatus.PENDING)

    # Scores par composant (0.0 - 1.0)
    score_global = Column(Float)
    score_logo = Column(Float)
    score_stitching = Column(Float)
    score_texture = Column(Float)
    score_label = Column(Float)
    score_rules = Column(Float)
    score_invoice = Column(Float)   # Score de vérification de la facture d'achat

    # Métadonnées pipeline
    pipeline_version = Column(String, default="v0.1")
    processing_time_ms = Column(Integer)   # Temps de traitement en ms
    gradcam_path = Column(String)           # Chemin vers le rapport Grad-CAM
    reviewed_by = Column(String)            # ID expert si vérif. humaine
    created_at = Column(DateTime, default=datetime.utcnow)

    article = relationship("Article", back_populates="verification")


# ════════════════════════════════════
# TRANSACTIONS (LOCATIONS)
# ════════════════════════════════════
class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    renter_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(Enum(TransactionStatus), default=TransactionStatus.PENDING)

    date_start = Column(DateTime, nullable=False)
    date_end = Column(DateTime, nullable=False)
    nb_days = Column(Integer, nullable=False)
    price_per_day = Column(Float, nullable=False)
    insurance_fee = Column(Float, default=25.0)
    total_amount = Column(Float, nullable=False)
    commission = Column(Float)       # 20% du total
    guarantee_amount = Column(Float)  # Caution bloquée

    stripe_payment_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relations
    article = relationship("Article", back_populates="transactions")
    renter = relationship("User", back_populates="rentals", foreign_keys=[renter_id])
    return_record = relationship("ReturnRecord", back_populates="transaction", uselist=False)


# ════════════════════════════════════
# RETOURS
# ════════════════════════════════════
class ReturnRecord(Base):
    __tablename__ = "return_records"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)

    # Checklist 4 points
    general_condition = Column(Boolean)    # État général OK ?
    cleanliness = Column(Boolean)          # Propreté OK ?
    hardware_intact = Column(Boolean)      # Quincaillerie OK ?
    accessories_returned = Column(Boolean) # Accessoires rendus ?

    condition = Column(Enum(ReturnCondition), default=ReturnCondition.PENDING_REVIEW)
    notes = Column(Text)
    guarantee_released = Column(Boolean, default=False)
    guarantee_deducted = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    transaction = relationship("Transaction", back_populates="return_record")


# ════════════════════════════════════
# JOURNAL D'ADMINISTRATION
# ════════════════════════════════════
class AdminAction(Base):
    """
    Trace chaque action prise depuis la console d'administration :
    qui (admin_id), quoi (action), sur quoi (cible_type/cible_id), et le
    détail (motif de rejet, ancien/nouveau statut, etc.).
    """
    __tablename__ = "admin_actions"

    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)       # ex: "archive_article", "suspend_user"
    cible_type = Column(String, nullable=False)   # ex: "article", "user", "transaction"
    cible_id = Column(Integer, nullable=True)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
