"""fixing the photo model

Revision ID: 1bb53eaa10ed
Revises: 59cd34d78697
Create Date: 2025-11-30 02:40:47.708175

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '1bb53eaa10ed'
down_revision = '59cd34d78697'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # safer ALTER for 'exif' (text -> jsonb), convert empty strings to NULL
    op.alter_column(
        'photos',
        'exif',
        existing_type=sa.VARCHAR(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
        postgresql_using="CASE WHEN trim(exif) = '' THEN NULL ELSE exif::jsonb END"
    )

    # safer ALTER for 'taken_at' (varchar -> timestamptz): empty -> NULL else cast
    # NOTE: This assumes your stored timestamps are ISO-8601 or Postgres-parseable strings.
    op.alter_column(
        'photos',
        'taken_at',
        existing_type=sa.VARCHAR(),
        type_=sa.DateTime(timezone=True),
        existing_nullable=True,
        postgresql_using="CASE WHEN trim(taken_at) = '' THEN NULL ELSE taken_at::timestamptz END"
    )

    # safer ALTER for 'extra_data' (text -> jsonb)
    op.alter_column(
        'photos',
        'extra_data',
        existing_type=sa.VARCHAR(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
        postgresql_using="CASE WHEN trim(extra_data) = '' THEN NULL ELSE extra_data::jsonb END"
    )


def downgrade() -> None:
    # Revert: jsonb -> varchar (use ::text for jsonb)
    op.alter_column(
        'photos',
        'extra_data',
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.VARCHAR(),
        existing_nullable=True,
        postgresql_using="extra_data::text"
    )
    op.alter_column(
        'photos',
        'taken_at',
        existing_type=sa.DateTime(timezone=True),
        type_=sa.VARCHAR(),
        existing_nullable=True,
        postgresql_using="taken_at::text"
    )
    op.alter_column(
        'photos',
        'exif',
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.VARCHAR(),
        existing_nullable=True,
        postgresql_using="exif::text"
    )
