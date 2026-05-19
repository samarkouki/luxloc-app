"""Configuration de la base de données SQLAlchemy"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./luxloc.db"
# En production : "postgresql://user:password@localhost/luxloc"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dépendance FastAPI pour injecter la session DB"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
