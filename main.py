"""
LUXLOC — Application Intégrée (Backend + Frontend)
Lance avec : py -m uvicorn main:app --reload
Ouvre : http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import articles, users, verification, transactions, returns
from app.database import engine, Base, SessionLocal
from app.models.models import *
import hashlib
from datetime import datetime, timedelta

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="LUXLOC API",
    description="API de la plateforme de location d'articles de luxe avec vérification IA d'authenticité",
    version="0.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(users.router, prefix="/api/v1/users", tags=["Utilisateurs"])
app.include_router(articles.router, prefix="/api/v1/articles", tags=["Articles"])
app.include_router(verification.router, prefix="/api/v1/verification", tags=["Vérification IA"])
app.include_router(transactions.router, prefix="/api/v1/transactions", tags=["Transactions"])
app.include_router(returns.router, prefix="/api/v1/returns", tags=["Retours"])

app.mount("/static", StaticFiles(directory="static"), name="static")


def simple_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health", tags=["Root"])
def health():
    return {"status": "healthy"}


@app.post("/api/v1/seed", tags=["Root"])
def seed_demo_data():
    db = SessionLocal()
    try:
        if db.query(User).count() > 0:
            return {"message": "Données déjà présentes", "users": db.query(User).count(), "articles": db.query(Article).count()}

        users_data = [
            User(email="sophie@luxloc.fr", hashed_password=simple_hash("demo123"), first_name="Sophie", last_name="Martin", role=UserRole.OWNER, is_verified=True, trust_score=0.92),
            User(email="emma@luxloc.fr", hashed_password=simple_hash("demo123"), first_name="Emma", last_name="Dubois", role=UserRole.RENTER, is_verified=True, trust_score=0.88),
            User(email="lucas@luxloc.fr", hashed_password=simple_hash("demo123"), first_name="Lucas", last_name="Bernard", role=UserRole.BOTH, is_verified=True, trust_score=0.95),
        ]
        db.add_all(users_data)
        db.flush()

        articles_data = [
            Article(owner_id=1, title="Speedy Bandoulière 25", brand="Louis Vuitton", model="Speedy 25", category=ArticleCategory.BAGS, description="Toile Monogram, cuir naturel. État impeccable, porté 3 fois.", price_per_day=85, retail_price=1200, condition="Excellent", is_available=True),
            Article(owner_id=1, title="Marmont Small Shoulder", brand="Gucci", model="Marmont GG", category=ArticleCategory.BAGS, description="Cuir matelassé noir, quincaillerie dorée.", price_per_day=65, retail_price=980, condition="Très bon", is_available=True),
            Article(owner_id=3, title="Submariner Date 41mm", brand="Rolex", model="126610LN", category=ArticleCategory.WATCHES, description="Acier Oystersteel, lunette Cerachrom noire. Révision 2024.", price_per_day=220, retail_price=9500, condition="Excellent", is_available=True),
            Article(owner_id=3, title="Kelly 28 Retourné", brand="Hermès", model="Kelly 28 Togo", category=ArticleCategory.BAGS, description="Cuir Togo Gold, quincaillerie Palladium. Livré avec dustbag et boîte.", price_per_day=195, retail_price=8500, condition="Excellent", is_available=True),
            Article(owner_id=1, title="Collier Vintage Alhambra", brand="Van Cleef & Arpels", model="Alhambra 10 motifs", category=ArticleCategory.JEWELRY, description="Or jaune 18K, nacre blanche. Certificat inclus.", price_per_day=110, retail_price=4800, condition="Neuf", is_available=True),
            Article(owner_id=3, title="Trench Heritage", brand="Burberry", model="Heritage Classic", category=ArticleCategory.CLOTHING, description="Gabardine de coton miel, taille 38.", price_per_day=55, retail_price=1800, condition="Bon", is_available=False),
            Article(owner_id=1, title="Aviator Classic", brand="Ray-Ban", model="RB3025", category=ArticleCategory.EYEWEAR, description="Monture or, verres verts G-15.", price_per_day=15, retail_price=180, condition="Excellent", is_available=True),
            Article(owner_id=3, title="Santos de Cartier", brand="Cartier", model="Santos Medium", category=ArticleCategory.WATCHES, description="Acier et or jaune 18K. Bracelet interchangeable.", price_per_day=180, retail_price=7200, condition="Excellent", is_available=True),
        ]
        db.add_all(articles_data)
        db.flush()

        photo_types = ["overview", "logo", "label", "stitching"]
        for art in articles_data:
            for pt in photo_types:
                db.add(ArticlePhoto(article_id=art.id, photo_type=pt, file_path=f"gs://luxloc-photos/{art.id}/{pt}.jpg"))

        import random
        for art in articles_data:
            if art.is_available:
                sg = round(0.85 + random.uniform(0, 0.12), 4)
                db.add(Verification(
                    article_id=art.id, status=VerificationStatus.VERIFIED,
                    score_global=sg,
                    score_logo=round(min(sg + random.uniform(-0.03, 0.05), 1.0), 4),
                    score_stitching=round(min(sg + random.uniform(-0.05, 0.03), 1.0), 4),
                    score_texture=round(min(sg + random.uniform(-0.02, 0.04), 1.0), 4),
                    score_label=round(min(sg + random.uniform(-0.06, 0.02), 1.0), 4),
                    score_rules=round(min(sg + random.uniform(-0.01, 0.05), 1.0), 4),
                    pipeline_version="v0.1-poc", processing_time_ms=random.randint(8000, 14000),
                ))

        db.add(Transaction(
            article_id=4, renter_id=2, status=TransactionStatus.ACTIVE,
            date_start=datetime.utcnow(), date_end=datetime.utcnow() + timedelta(days=3),
            nb_days=3, price_per_day=195, insurance_fee=25, total_amount=610,
            commission=122, guarantee_amount=975,
        ))

        db.commit()
        return {"message": "Données de démo injectées !", "users": 3, "articles": len(articles_data)}
    finally:
        db.close()
