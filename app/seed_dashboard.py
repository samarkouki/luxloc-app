"""
app/seed_dashboard.py

Génère des données de démo RICHES pour alimenter le dashboard admin.

À utiliser APRÈS le seed de base (POST /api/v1/seed), qui crée les 3
premiers utilisateurs et les 18 articles. Ce script ajoute :

- Des utilisateurs supplémentaires (owners/renters), avec une date
  d'inscription étalée dans le temps → pour une courbe de croissance
  utilisateurs crédible.
- Un historique de transactions réparti sur plusieurs mois, avec des
  statuts variés (completed, active, confirmed, cancelled) → pour des
  graphiques de revenus, de taux d'annulation, de saisonnalité.
- Des retours (ReturnRecord) pour les locations terminées, avec une
  répartition réaliste conforme / à examiner / endommagé.

Il est conçu pour être rejoué : il AJOUTE des données à chaque appel,
il ne supprime rien. Pour repartir de zéro : POST /api/v1/reset-and-seed
puis relancer /api/v1/seed et /api/v1/seed-dashboard.
"""
import random
import hashlib
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.models import (
    User, UserRole, Article,
    Transaction, TransactionStatus, ReturnRecord, ReturnCondition,
)

CONDITION_SCALE = ["Neuf", "Excellent", "Très bon", "Bon", "Correct", "Endommagé"]


def _degrade_condition(cond: str, steps: int = 1) -> str:
    idx = CONDITION_SCALE.index(cond) if cond in CONDITION_SCALE else 1
    return CONDITION_SCALE[min(idx + steps, len(CONDITION_SCALE) - 1)]

FIRST_NAMES = ["Alice", "Nora", "Yanis", "Camille", "Karim", "Julie", "Adam",
               "Lina", "Hugo", "Sarah", "Nathan", "Chloe", "Rayan", "Manon",
               "Leo", "Ines", "Sami", "Elise", "Mehdi", "Zoe", "Farid",
               "Clara", "Bilal", "Anaïs", "Yasmine", "Noe", "Selma", "Theo"]
LAST_NAMES = ["Benali", "Martin", "Dupont", "Moreau", "Haddad", "Girard",
              "Rousseau", "Fontaine", "Aziz", "Lambert", "Roche", "Faure",
              "Meyer", "Perrin", "Chevalier", "Bouzid"]


def hash_pw(pw: str) -> str:
    # ATTENTION : SHA256 nu, cohérent avec le reste du projet pour
    # l'instant. À remplacer par bcrypt (passlib) quand l'auth sera
    # implémentée — voir la note de sécurité séparée.
    return hashlib.sha256(pw.encode()).hexdigest()


def _random_name():
    return random.choice(FIRST_NAMES), random.choice(LAST_NAMES)


def seed_dashboard_data(months_back: int = 12, n_extra_users: int = 24,
                         n_transactions: int = 220) -> dict:
    db: Session = SessionLocal()
    try:
        # ── 1. Utilisateurs supplémentaires ─────────────────────────
        existing_emails = {e for (e,) in db.query(User.email).all()}
        new_users = []
        for i in range(n_extra_users):
            fn, ln = _random_name()
            email = f"{fn.lower()}.{ln.lower()}{i}@luxloc-demo.fr"
            if email in existing_emails:
                continue
            role = random.choices(
                [UserRole.RENTER, UserRole.OWNER, UserRole.BOTH],
                weights=[0.55, 0.25, 0.20],
            )[0]
            user = User(
                email=email,
                hashed_password=hash_pw("demo123"),
                first_name=fn, last_name=ln, role=role,
                is_verified=random.random() > 0.08,
                trust_score=round(random.uniform(0.55, 0.99), 2),
            )
            db.add(user)
            new_users.append(user)
        db.flush()

        all_users = db.query(User).filter(User.role != UserRole.ADMIN).all()
        owners = [u for u in all_users if u.role in (UserRole.OWNER, UserRole.BOTH)]
        renters = [u for u in all_users if u.role in (UserRole.RENTER, UserRole.BOTH)]

        articles = db.query(Article).all()
        if not articles:
            return {"error": "Aucun article en base — lancez d'abord POST /api/v1/seed"}

        # Répartir aléatoirement une partie des articles sur les nouveaux
        # owners, pour varier les stats par propriétaire dans le dashboard.
        for art in articles:
            if owners and random.random() < 0.3:
                art.owner_id = random.choice(owners).id

        # ── 2. Transactions historiques ──────────────────────────────
        statuses = [TransactionStatus.COMPLETED, TransactionStatus.ACTIVE,
                    TransactionStatus.CONFIRMED, TransactionStatus.CANCELLED]
        weights = [0.70, 0.10, 0.08, 0.12]

        created_tx = 0
        for _ in range(n_transactions):
            article = random.choice(articles)
            possible_renters = [r for r in renters if r.id != article.owner_id]
            if not possible_renters:
                continue
            renter = random.choice(possible_renters)

            days_ago = random.randint(1, months_back * 30)
            date_start = datetime.utcnow() - timedelta(days=days_ago)
            nb_days = random.randint(2, 14)
            date_end = date_start + timedelta(days=nb_days)

            status = random.choices(statuses, weights=weights)[0]
            price_per_day = article.price_per_day
            subtotal = price_per_day * nb_days
            insurance = 25.0
            total = subtotal + insurance
            commission = round(total * 0.20, 2)
            guarantee = round(price_per_day * 5, 2)

            tx = Transaction(
                article_id=article.id,
                renter_id=renter.id,
                status=status,
                date_start=date_start,
                date_end=date_end,
                nb_days=nb_days,
                price_per_day=price_per_day,
                insurance_fee=insurance,
                total_amount=round(total, 2),
                commission=commission,
                guarantee_amount=guarantee,
                created_at=date_start,
            )
            db.add(tx)
            db.flush()
            created_tx += 1

            # Retour pour les transactions terminées uniquement
            if status == TransactionStatus.COMPLETED:
                roll = random.random()
                if roll < 0.82:
                    condition, deducted = ReturnCondition.CONFORM, 0.0
                elif roll < 0.94:
                    condition, deducted = ReturnCondition.PENDING_REVIEW, 0.0
                else:
                    condition, deducted = ReturnCondition.DAMAGED, round(guarantee * 0.5, 2)

                # Ponctualité : un retour endommagé/à examiner traîne un peu
                # plus souvent — sinon la majorité rentre à temps ou en avance.
                late_prob = 0.15 if condition == ReturnCondition.CONFORM else 0.35
                if random.random() < late_prob:
                    actual_return_date = date_end + timedelta(days=random.randint(1, 6))
                else:
                    actual_return_date = date_end - timedelta(days=random.randint(0, 2))
                actual_return_date = min(actual_return_date, datetime.utcnow())
                returned_on_time = actual_return_date <= date_end

                condition_at_rental = article.condition or "Excellent"
                if condition == ReturnCondition.DAMAGED:
                    condition_at_return = "Endommagé"
                elif condition == ReturnCondition.PENDING_REVIEW:
                    condition_at_return = _degrade_condition(condition_at_rental, 1)
                else:
                    condition_at_return = condition_at_rental

                db.add(ReturnRecord(
                    transaction_id=tx.id,
                    general_condition=condition != ReturnCondition.DAMAGED,
                    cleanliness=condition != ReturnCondition.DAMAGED,
                    hardware_intact=condition != ReturnCondition.DAMAGED,
                    accessories_returned=True,
                    condition=condition,
                    notes="Généré automatiquement (simulation dashboard)",
                    guarantee_released=(condition == ReturnCondition.CONFORM),
                    guarantee_deducted=deducted,
                    condition_at_rental=condition_at_rental,
                    condition_at_return=condition_at_return,
                    actual_return_date=actual_return_date,
                    returned_on_time=returned_on_time,
                    created_at=actual_return_date,
                ))

        db.commit()

        return {
            "message": "Données de simulation injectées pour le dashboard",
            "nouveaux_utilisateurs": len(new_users),
            "transactions_creees": created_tx,
            "periode_simulee_mois": months_back,
            "total_utilisateurs": db.query(User).count(),
            "total_transactions": db.query(Transaction).count(),
        }
    finally:
        db.close()


if __name__ == "__main__":
    print(seed_dashboard_data())