"""Add authentication tables (manual)

Revision ID: 4df082f55528
Revises: 402fae8fb853
Create Date: 2025-11-28 04:30:51.330028

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '4df082f55528'
down_revision = '402fae8fb853'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Tables already created manually via SQLAlchemy
    # This migration just tracks the change in Alembic history
    pass


def downgrade() -> None:
    op.drop_table('login_history')
    op.drop_table('refresh_tokens')
    op.drop_table('otps')
