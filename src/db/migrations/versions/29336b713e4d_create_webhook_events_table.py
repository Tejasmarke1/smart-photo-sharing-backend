"""create webhook_events table

Revision ID: 29336b713e4d
Revises: 485b00555561
Create Date: 2025-12-30 20:26:32.550887

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '29336b713e4d'
down_revision = '485b00555561'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'webhook_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('provider', sa.String(100), nullable=False),
        sa.Column('event_id', sa.String(255), nullable=False, unique=True),
        sa.Column('event_type', sa.String(100), nullable=False),

        sa.Column('resource_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('resource_type', sa.String(100), nullable=True),

        sa.Column('payload', postgresql.JSON(), nullable=False),

        sa.Column('processed', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('received_at', sa.DateTime(), nullable=False),

        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_webhook_events_event_id', 'webhook_events', ['event_id'], unique=True)
    op.create_index('ix_webhook_events_provider', 'webhook_events', ['provider'])
    op.create_index('ix_webhook_events_event_type', 'webhook_events', ['event_type'])
    op.create_index('ix_webhook_events_processed', 'webhook_events', ['processed'])


def downgrade() -> None:
    op.drop_index('ix_webhook_events_event_id', table_name= 'webhook_events')
    op.drop_index('ix_webhook_events_provider', table_name= 'webhook_events')
    op.drop_index('ix_webhook_events_event_type', table_name= 'webhook_events')
    op.drop_index('ix_webhook_events_processed', table_name= 'webhook_events')
    op.drop_table('webhook_events')