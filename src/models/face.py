"""Face model."""
from sqlalchemy import Column, String, Float, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
import uuid

# FIXED: import Base from db.base (not src.db.base)
from db.base import Base
from .base import TimestampMixin


class Face(Base, TimestampMixin):
    """Detected face instance."""
    
    __tablename__ = 'faces'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    photo_id = Column(UUID(as_uuid=True), ForeignKey('photos.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Bounding box (stored as JSONB string: {x, y, w, h})
    bbox = Column(String, nullable=False)
    
    # Face embedding (512-dimensional vector)
    embedding = Column(Vector(512), nullable=True)
    
    # Detection confidence
    confidence = Column(Float, nullable=False)
    
    # Face thumbnail
    thumbnail_s3_key = Column(String(512), nullable=True)
    
    # Quality scores
    blur_score = Column(Float, nullable=True)
    brightness_score = Column(Float, nullable=True)
    
    # Additional metadata
    extra_data = Column(String, nullable=True)  # JSONB
    
    # Relationships
    photo = relationship('Photo', back_populates='faces')
    person_mapping = relationship('FacePerson', back_populates='face', uselist=False, cascade='all, delete-orphan')
    
    @property
    def person(self):
        """Get associated person if labeled."""
        return self.person_mapping.person if self.person_mapping else None
    
    def __repr__(self) -> str:
        return f'<Face(id={self.id}, confidence={self.confidence})>'
