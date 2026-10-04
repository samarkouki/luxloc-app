"""Schémas Pydantic — Validation des requêtes et réponses API"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ════════════════════════════════════
# ENUMS
# ════════════════════════════════════
class CategoryEnum(str, Enum):
    bags = "bags"
    watches = "watches"
    jewelry = "jewelry"
    clothing = "clothing"
    eyewear = "eyewear"


# ════════════════════════════════════
# USERS
# ════════════════════════════════════
class UserCreate(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    phone: Optional[str] = None
    role: str = "both"

class UserResponse(BaseModel):
    id: int
    email: str
    first_name: str
    last_name: str
    role: str
    is_verified: bool
    is_active: bool = True
    trust_score: float
    created_at: datetime

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    email: str
    password: str

class PasswordForgotRequest(BaseModel):
    email: str

class PasswordResetRequest(BaseModel):
    token: str = Field(..., min_length=10)
    new_password: str = Field(..., min_length=8, max_length=128)

class UserAdminUpdate(BaseModel):
    """Modification d'un compte par un administrateur — tous les champs optionnels."""
    role: Optional[str] = None
    is_active: Optional[bool] = None


# ════════════════════════════════════
# ARTICLES
# ════════════════════════════════════
class ArticleCreate(BaseModel):
    title: str
    brand: str
    model: Optional[str] = None
    category: CategoryEnum
    description: Optional[str] = None
    price_per_day: float = Field(gt=0)
    retail_price: Optional[float] = None
    condition: str = "Excellent"

class ArticleResponse(BaseModel):
    id: int
    owner_id: int
    title: str
    brand: str
    model: Optional[str]
    category: str
    description: Optional[str]
    price_per_day: float
    condition: str
    is_available: bool
    verification_status: Optional[str] = None
    verification_score: Optional[float] = None
    photo: Optional[str] = None  # Photo principale (overview) en base64
    created_at: datetime

    class Config:
        from_attributes = True

class ArticleFilter(BaseModel):
    category: Optional[CategoryEnum] = None
    brand: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    verified_only: bool = True

class ArticleUpdate(BaseModel):
    """Modification d'un article par un administrateur — tous les champs optionnels."""
    title: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    category: Optional[CategoryEnum] = None
    price_per_day: Optional[float] = Field(default=None, gt=0)
    condition: Optional[str] = None
    is_available: Optional[bool] = None


# ════════════════════════════════════
# VÉRIFICATION
# ════════════════════════════════════
class VerificationRequest(BaseModel):
    article_id: int
    photo_ids: List[int] = Field(min_length=3, max_length=5)
    invoice_ocr_text: Optional[str] = None  # Texte OCR de la facture (Google Vision en prod)

class VerificationScores(BaseModel):
    score_global: float
    score_logo: float
    score_stitching: float
    score_texture: float
    score_label: float
    score_rules: float

class ExpertRejectRequest(BaseModel):
    motif: str = Field(min_length=1, max_length=300)

class VerificationResponse(BaseModel):
    id: int
    article_id: int
    status: str
    scores: VerificationScores
    pipeline_version: str
    processing_time_ms: int
    gradcam_path: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ════════════════════════════════════
# TRANSACTIONS
# ════════════════════════════════════
class TransactionCreate(BaseModel):
    article_id: int
    date_start: datetime
    date_end: datetime

class TransactionResponse(BaseModel):
    id: int
    article_id: int
    renter_id: int
    status: str
    date_start: datetime
    date_end: datetime
    nb_days: int
    price_per_day: float
    insurance_fee: float
    total_amount: float
    commission: float
    created_at: datetime

    class Config:
        from_attributes = True

class TransactionAdminUpdate(BaseModel):
    """Changement de statut d'une location par un administrateur."""
    status: str

class PriceEstimate(BaseModel):
    nb_days: int
    price_per_day: float
    subtotal: float
    insurance_fee: float
    total: float
    commission: float
    owner_receives: float


# ════════════════════════════════════
# RETOURS
# ════════════════════════════════════
class ReturnCreate(BaseModel):
    transaction_id: int
    general_condition: bool
    cleanliness: bool
    hardware_intact: bool
    accessories_returned: bool
    notes: Optional[str] = None

class ReturnResponse(BaseModel):
    id: int
    transaction_id: int
    general_condition: bool
    cleanliness: bool
    hardware_intact: bool
    accessories_returned: bool
    condition: str
    guarantee_released: bool
    guarantee_deducted: float
    condition_at_rental: Optional[str] = None
    condition_at_return: Optional[str] = None
    actual_return_date: Optional[datetime] = None
    returned_on_time: Optional[bool] = None
    created_at: datetime

    class Config:
        from_attributes = True
