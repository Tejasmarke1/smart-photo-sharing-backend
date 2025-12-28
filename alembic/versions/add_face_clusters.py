"""Add face_clusters table for clustering workflow

Revision ID: add_face_clusters
Revises: [previous_revision]
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_face_clusters'
down_revision = '[previous_revision]'  # Update with actual previous revision
branch_labels = None
depends_on = None


def upgrade():
    # Create face_clusters table
    op.create_table(
        'face_clusters',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('album_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('albums.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('job_id', sa.String(255), nullable=True, index=True),
        sa.Column('cluster_label', sa.Integer(), nullable=False, index=True),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('avg_similarity', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='pending'),
        sa.Column('representative_face_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('faces.id', ondelete='SET NULL'), nullable=True),
        sa.Column('person_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('persons.id', ondelete='SET NULL'), nullable=True),
        sa.Column('merged_into_cluster_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('face_clusters.id', ondelete='SET NULL'), nullable=True),
        sa.Column('reviewed_by_user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('review_notes', sa.String(1000), nullable=True),
        sa.Column('face_ids', postgresql.JSON(), nullable=False),
        sa.Column('extra_data', postgresql.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    
    # Create indexes for better query performance
    op.create_index('ix_face_clusters_album_id', 'face_clusters', ['album_id'])
    op.create_index('ix_face_clusters_job_id', 'face_clusters', ['job_id'])
    op.create_index('ix_face_clusters_status', 'face_clusters', ['status'])
    op.create_index('ix_face_clusters_cluster_label', 'face_clusters', ['cluster_label'])


def downgrade():
    op.drop_table('face_clusters')
