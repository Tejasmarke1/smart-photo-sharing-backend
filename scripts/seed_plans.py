from uuid import uuid4
from sqlalchemy.orm import Session

from src.db.base import SessionLocal
from src.models.plan import Plan


PLANS = [
    # -------------------------
    # Photographer Plans
    # -------------------------
    {
        "code": "photographer_free_5gb",
        "name": "Photographer Free",
        "role": "photographer",
        "storage_limit_bytes": 5 * 1024**3,
        "price_cents": 0,
        "currency": "INR",
        "billing_cycle": "yearly",
    },
    {
        "code": "photographer_standard_100gb",
        "name": "Photographer Standard",
        "role": "photographer",
        "storage_limit_bytes": 100 * 1024**3,
        "price_cents": 749000,
        "currency": "INR",
        "billing_cycle": "yearly",
    },
    {
        "code": "photographer_essential_500gb",
        "name": "Photographer Essential",
        "role": "photographer",
        "storage_limit_bytes": 500 * 1024**3,
        "price_cents": 1499000,
        "currency": "INR",
        "billing_cycle": "yearly",
    },

    # -------------------------
    # User Plans
    # -------------------------
    {
        "code": "user_free_2gb",
        "name": "User Free",
        "role": "user",
        "storage_limit_bytes": 2 * 1024**3,
        "price_cents": 0,
        "currency": "INR",
        "billing_cycle": "yearly",
    },
    {
        "code": "user_standard_50gb",
        "name": "User Standard",
        "role": "user",
        "storage_limit_bytes": 50 * 1024**3,
        "price_cents": 149000,
        "currency": "INR",
        "billing_cycle": "yearly",
    },
    {
        "code": "user_essential_200gb",
        "name": "User Essential",
        "role": "user",
        "storage_limit_bytes": 200 * 1024**3,
        "price_cents": 299000,
        "currency": "INR",
        "billing_cycle": "yearly",
    },
]


def seed_plans():
    db: Session = SessionLocal()

    for plan_data in PLANS:
        existing = (
            db.query(Plan)
            .filter(Plan.code == plan_data["code"])
            .first()
        )

        if existing:
            print(f"✔ Plan already exists: {plan_data['code']}")
            continue

        plan = Plan(
            id=uuid4(),
            **plan_data,
            is_active=True,
        )

        db.add(plan)
        print(f"➕ Created plan: {plan_data['code']}")

    db.commit()
    db.close()


if __name__ == "__main__":
    seed_plans()
