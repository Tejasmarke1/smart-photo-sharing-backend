#!/usr/bin/env python
"""Seed database with test data."""
import sys
from pathlib import Path
import hashlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sqlalchemy.orm import Session
from src.db.base import SessionLocal, engine, Base
from src.models import User, Album, UserRole


def hash_password(password: str) -> str:
    """Simple hash for development (use proper bcrypt in production)."""
    return hashlib.sha256(password.encode()).hexdigest()


def seed_users(db: Session):
    """Create test users."""
    users = [
        User(
            name="Test Photographer",
            email="photographer@test.com",
            phone="+919876543210",
            hashed_password=hash_password("password123"),
            role=UserRole.photographer,
            is_active=True,
            is_verified=True,
        ),
        User(
            name="Test Guest",
            email="guest@test.com",
            phone="+919876543211",
            hashed_password=hash_password("password123"),
            role=UserRole.guest,
            is_active=True,
            is_verified=False,
        ),
        User(
            name="Admin User",
            email="admin@test.com",
            phone="+919876543212",
            hashed_password=hash_password("admin123"),
            role=UserRole.admin,
            is_active=True,
            is_verified=True,
        ),
    ]
    
    for user in users:
        existing = db.query(User).filter(User.email == user.email).first()
        if not existing:
            db.add(user)
            print(f"  → Created user: {user.email} (role: {user.role})")
    
    db.commit()
    print("✅ Created test users")


def seed_albums(db: Session):
    """Create test albums."""
    photographer = db.query(User).filter(
        User.role == UserRole.photographer
    ).first()
    
    if photographer:
        albums = [
            Album(
                photographer_id=photographer.id,
                title="Test Wedding Event",
                description="Beautiful wedding in Mumbai",
                location="Mumbai, India",
                is_public=True,
                face_detection_enabled=True,
                watermark_enabled=True,
                download_enabled=True,
            ),
            Album(
                photographer_id=photographer.id,
                title="Birthday Party 2024",
                description="Kids birthday celebration",
                location="Delhi, India",
                is_public=True,
                face_detection_enabled=True,
            ),
        ]
        
        for album in albums:
            existing = db.query(Album).filter(Album.title == album.title).first()
            if not existing:
                db.add(album)
                print(f"  → Created album: {album.title}")
        
        db.commit()
        print("✅ Created test albums")


if __name__ == "__main__":
    print("🌱 Seeding database...")
    print("=" * 50)
    
    db = SessionLocal()
    try:
        seed_users(db)
        seed_albums(db)
        print("=" * 50)
        print("✅ Database seeded successfully!")
        print("\n📝 Test Credentials:")
        print("   Photographer: photographer@test.com / password123")
        print("   Guest: guest@test.com / password123")
        print("   Admin: admin@test.com / admin123")
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
