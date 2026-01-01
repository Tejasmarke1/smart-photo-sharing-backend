"""add plan_id to subscriptions

Revision ID: e68063e8d72d
Revises: 8bbdf0a30e81
Create Date: 2025-12-30 19:56:32.195887

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid


# revision identifiers, used by Alembic.
revision = 'e68063e8d72d'
down_revision = '8bbdf0a30e81'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'subscriptions',
        sa.Column('plan_id', postgresql.UUID(as_uuid=True), nullable=True)
    )

    op.create_foreign_key(
        'fk_subscriptions_plan',
        'subscriptions',
        'plans',
        ['plan_id'],
        ['id'],
        ondelete='RESTRICT'
    )


def downgrade() -> None:
    pass
