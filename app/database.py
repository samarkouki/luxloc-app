"""
Configuration de la base de données SQLAlchemy.

— En local : SQLite (fichier luxloc.db dans le dossier courant).
— En production sur Railway : on lit la variable d'environnement
  DATABASE_URL (renseignée automatiquement par le plugin PostgreSQL).
  Les données persistent entre les redéploiements.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Railway peut fournir DATABASE_URL au format postgres:// (héritage Heroku),
# mais SQLAlchemy 2.x exige postgresql://. On corrige automatiquement.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./luxloc.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLite a besoin d'un argument supplémentaire pour fonctionner en multi-thread ;
# pas PostgreSQL.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dépendance FastAPI pour injecter la session DB."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
