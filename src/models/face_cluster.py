"""Face Cluster model for tracking clustering results."""
from sqlalchemy import Column, String, Integer, Float, ForeignKey, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin


class FaceCluster(Base, TimestampMixin):
    """
    Face clustering results before being confirmed as persons.
    
    Workflow:
    1. Clustering algorithm creates clusters
    2. User reviews clusters
    3. Accepted clusters -> Person records
    4. Rejected clusters -> marked as noise
    """
    
    __tablename__ = 'face_clusters'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    album_id = Column(UUID(as_uuid=True), ForeignKey('albums.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Clustering job reference
    job_id = Column(String(255), nullable=True, index=True)
    
    # Cluster metadata
    cluster_label = Column(Integer, nullable=False, index=True)  # Cluster ID from algorithm
    size = Column(Integer, nullable=False)  # Number of faces
    
    # Quality metrics
    avg_similarity = Column(Float, nullable=True)  # Average intra-cluster similarity
    confidence_score = Column(Float, nullable=True)  # Cluster quality score
    
    # Status
    status = Column(String(50), nullable=False, default='pending')  # pending, accepted, rejected, split, merged
    
    # Representative face
    representative_face_id = Column(UUID(as_uuid=True), ForeignKey('faces.id', ondelete='SET NULL'), nullable=True)
    
    # If accepted, link to person
    person_id = Column(UUID(as_uuid=True), ForeignKey('persons.id', ondelete='SET NULL'), nullable=True)
    
    # If merged, track parent cluster
    merged_into_cluster_id = Column(UUID(as_uuid=True), ForeignKey('face_clusters.id', ondelete='SET NULL'), nullable=True)
    
    # Review metadata
    reviewed_by_user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    review_notes = Column(String(1000), nullable=True)
    
    # Face IDs in this cluster (stored as JSON array)
    face_ids = Column(JSON, nullable=False)
    
    # Additional metadata
    extra_data = Column(JSON, nullable=True)
    
    # Relationships
    album = relationship('Album', backref='face_clusters')
    representative_face = relationship('Face', foreign_keys=[representative_face_id])
    person = relationship('Person', foreign_keys=[person_id])
    reviewed_by = relationship('User', foreign_keys=[reviewed_by_user_id])
    merged_into = relationship('FaceCluster', remote_side=[id], foreign_keys=[merged_into_cluster_id])
    
    def __repr__(self) -> str:
        return f'<FaceCluster(id={self.id}, label={self.cluster_label}, size={self.size}, status={self.status})>'
