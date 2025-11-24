#!/usr/bin/env python
import sys
from pathlib import Path

# Ensure repo src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.config import settings
print(f"DB URL: {settings.DATABASE_URL}")

from db.base import Base
print("Base loaded")

# import models (adjust names if your models module path differs)
from models import User, Album, Photo, Face, Person, Payment, Subscription, AuditLog, Download

print(f"Models loaded. Tables: {len(Base.metadata.tables)}")
print(f"Table names: {list(Base.metadata.tables.keys())}")
