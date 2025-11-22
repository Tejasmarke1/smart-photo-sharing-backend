#!/usr/bin/env python
\"\"\"Seed database with test data.\"\"\"
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from db.base import SessionLocal
from models.user import User

def seed_users():
    \"\"\"Create test users.\"\"\"
    db = SessionLocal()
    try:
        # Add test photographer
        photographer = User(
            name="Test Photographer",
            email="photographer@test.com",
            phone="+919876543210",
            role="photographer"
        )
        db.add(photographer)
        db.commit()
        print("? Created test users")
    finally:
        db.close()

if __name__ == "__main__":
    print("?? Seeding database...")
    seed_users()
    print("? Database seeded successfully!")
