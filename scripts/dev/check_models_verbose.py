#!/usr/bin/env python
"""
scripts/dev/check_models_verbose.py

Attempts to import each model individually and reports detailed errors.
Run: poetry run python scripts/dev/check_models_verbose.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.config import settings
print(f"DB URL: {settings.DATABASE_URL}")

from db.base import Base
print("Base loaded")
print()

models_to_test = {
    "models": "import models; print('models imported')",
    "models.user": "from models.user import User; print('User OK')",
    "models.album": "from models.album import Album; print('Album OK')",
    "models.photo": "from models.photo import Photo; print('Photo OK')",
    "models.face": "from models.face import Face; print('Face OK')",
    "models.person": "from models.person import Person; print('Person OK')",
    "models.payment": "from models.payment import Payment; print('Payment OK')",
    "models.subscription": "from models.subscription import Subscription; print('Subscription OK')",
    "models.audit_log": "from models.audit_log import AuditLog; print('AuditLog OK')",
    "models.download": "from models.download import Download; print('Download OK')",
}

errors = []
for name, stmt in models_to_test.items():
    print(f"--- Testing {name} ---")
    try:
        exec(stmt, {})
    except Exception as e:
        import traceback
        print(f"ERROR importing {name}: {e}")
        traceback.print_exc()
        errors.append((name, e))
    print()

print("Summary:")
if not errors:
    print("All model imports succeeded.")
    print(f"Tables available: {len(Base.metadata.tables)}")
    print(f"Table names: {list(Base.metadata.tables.keys())}")
else:
    print(f"{len(errors)} import errors detected:")
    for nm, err in errors:
        print(f" - {nm}: {err}")
    sys.exit(2)
