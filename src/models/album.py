"""Album model."""
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
import secrets

from src.db.base import Base
from .base import TimestampMixin, SoftDeleteMixin
from db.base import Base


class Album(Base, TimestampMixin, SoftDeleteMixin):
    """Album/Event model."""
    
    __tablename__ = 'albums'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    photographer_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Album details
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    
    # Sharing configuration
    sharing_code = Column(String(32), unique=True, nullable=False, index=True, default=lambda: secrets.token_urlsafe(16))
    is_public = Column(Boolean, default=False, nullable=False)
    consent_required = Column(Boolean, default=True, nullable=False)
    password_protected = Column(Boolean, default=False, nullable=False)
    album_password = Column(String(255), nullable=True)  # Hashed password
    
    # Feature flags
    face_detection_enabled = Column(Boolean, default=True, nullable=False)
    watermark_enabled = Column(Boolean, default=True, nullable=False)
    download_enabled = Column(Boolean, default=True, nullable=False)
    
    # Metadata
    cover_photo_url = Column(String(512), nullable=True)
    extra_data = Column(String, nullable=True)  # JSONB
    
    # Relationships
    photographer = relationship('User', back_populates='albums', foreign_keys=[photographer_id])
    photos = relationship('Photo', back_populates='album', cascade='all, delete-orphan')
    persons = relationship('Person', back_populates='album', cascade='all, delete-orphan')
    payments = relationship('Payment', back_populates='album')
    
    def __repr__(self) -> str:
        return f'<Album(id={self.id}, title={self.title})>'

