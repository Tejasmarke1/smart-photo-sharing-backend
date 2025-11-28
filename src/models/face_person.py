"""Face to Person mapping table."""
from sqlalchemy import Column, ForeignKey, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.db.base import Base
from .base import TimestampMixin



class FacePerson(Base, TimestampMixin):
    """Mapping table between faces and persons."""
    
    __tablename__ = 'face_person'
    
    face_id = Column(UUID(as_uuid=True), ForeignKey('faces.id', ondelete='CASCADE'), primary_key=True)
    person_id = Column(UUID(as_uuid=True), ForeignKey('persons.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Confidence score for this mapping (useful for auto-labeling)
    confidence = Column(Float, nullable=True)
    
    # Manual vs automatic assignment
    is_manual = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    face = relationship('Face', back_populates='person_mapping')
    person = relationship('Person', back_populates='face_mappings')
    
    def __repr__(self) -> str:
        return f'<FacePerson(face_id={self.face_id}, person_id={self.person_id})>'

