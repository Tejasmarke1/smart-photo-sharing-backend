import os
import sys
import importlib
import logging
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# --- project root and src on path (robust) ---
here = Path(__file__).resolve().parent
repo_root = here.parent.parent  # adjust if alembic dir depth differs; this points two levels up from env.py
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- optional: debug sys.path to see importability ---
logging.getLogger().debug("sys.path = %s", sys.path)

# import settings and Base
try:
    from src.app.config import settings
    from src.db.base import Base
except Exception as e:
    logging.exception("Failed to import app.config or db.base. Check PYTHONPATH and src layout. Error:")
    raise

# import model modules explicitly so Base.metadata is fully populated
# Replace or add module names for every module under src/models that defines models
model_modules = [
    "src.models.user",
    "src.models.face",
    "src.models.otp",
    "src.models.album",
    "src.models.payment",
    "src.models.login_history",
    "src.models.refresh_token",
    "src.models.subscription",
    "src.models.photo",
    "src.models.face_person",
    "src.models.audit_log",
    "src.models.download",
    "src.models.person",
    "src.models.face_cluster",
]

for mod in model_modules:
    try:
        importlib.import_module(mod)
    except Exception:
        logging.exception("Failed to import model module '%s'.", mod)
        # continue to raise so the error is visible to Alembic
        raise

# legacy wildcard import fallback (only if you have a top-level models/__init__.py that imports submodules)
# try:
#     import models  # noqa: F401
# except Exception:
#     logging.exception("Failed to import top-level 'models' package.")
#     raise

# Alembic config
config = context.config

# override DB URL from settings (safe replace for 'localhost' DNS issues)
db_url = settings.DATABASE_URL
# db_url = db_url.replace("localhost", "127.0.0.1")
config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
