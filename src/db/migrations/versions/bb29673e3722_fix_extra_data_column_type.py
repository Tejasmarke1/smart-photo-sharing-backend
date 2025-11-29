"""fix extra_data column type

Revision ID: bb29673e3722
Revises: abc0016d28a4
Create Date: 2025-11-29 17:22:09.689765

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = 'bb29673e3722'
down_revision = 'abc0016d28a4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Change extra_data column from String to JSONB
    op.alter_column(
        'albums',
        'extra_data',
        type_=postgresql.JSONB,
        existing_type=sa.String(),
        existing_nullable=True,
        postgresql_using='extra_data::jsonb'  # Convert existing data
    )


def downgrade() -> None:
    # Revert extra_data column from JSONB to String
    op.alter_column(
        'albums',
        'extra_data',
        type_=sa.String(),
        existing_type=postgresql.JSONB,
        existing_nullable=True,
        postgresql_using='extra_data::text'  # Convert back to text
    )
