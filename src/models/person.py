"""Person model for face clusters."""
from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin, SoftDeleteMixin
from db.base import Base


class Person(Base, TimestampMixin, SoftDeleteMixin):
    """Person cluster/label."""
    
    __tablename__ = 'persons'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    album_id = Column(UUID(as_uuid=True), ForeignKey('albums.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Person details
    name = Column(String(255), nullable=True)
    phone = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True)
    
    # Representative face
    representative_face_id = Column(UUID(as_uuid=True), ForeignKey('faces.id', ondelete='SET NULL'), nullable=True)
    
    # Additional metadata
    extra_data = Column(String, nullable=True)  # JSONB
    
    # Relationships
    album = relationship('Album', back_populates='persons')
    face_mappings = relationship('FacePerson', back_populates='person', cascade='all, delete-orphan')
    
    @property
    def face_count(self) -> int:
        """Get count of faces assigned to this person."""
        return len(self.face_mappings)
    
    def __repr__(self) -> str:
        return f'<Person(id={self.id}, name={self.name})>'

