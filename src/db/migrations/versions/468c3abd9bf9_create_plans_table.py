"""create plans table

Revision ID: 468c3abd9bf9
Revises: 81b2244cc55a
Create Date: 2025-12-30 19:50:16.561128

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '468c3abd9bf9'
down_revision = '81b2244cc55a'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'plans',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('code', sa.String(length=100), nullable=False, unique=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('storage_limit_bytes', sa.BigInteger(), nullable=False),
        sa.Column('price_cents', sa.BigInteger(), nullable=False),
        sa.Column('currency', sa.String(length=3), server_default='INR'),
        sa.Column('billing_cycle', sa.String(length=20), nullable=False),
        sa.Column('razorpay_plan_id', sa.String(length=255), unique=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('sort_order', sa.Integer()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    
    


def downgrade() -> None:
    op.drop_table('plans')
