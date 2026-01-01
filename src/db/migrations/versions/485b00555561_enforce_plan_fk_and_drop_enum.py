"""enforce plan fk and drop enum

Revision ID: 485b00555561
Revises: e68063e8d72d
Create Date: 2025-12-30 20:00:27.335857

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '485b00555561'
down_revision = 'e68063e8d72d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        'subscriptions',
        'plan_id',
        nullable=False
    )

    op.drop_column('subscriptions', 'plan')


def downgrade() -> None:
    op.add_column(
        'subscriptions',
        sa.Column(
            'plan',
            sa.Enum('FREE', 'STANDARD', 'ESSENTIAL', name='subscriptionplan'),
            nullable=False
        )
    )
