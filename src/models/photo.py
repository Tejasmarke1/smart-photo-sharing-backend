"""Photo model."""
from sqlalchemy import Column, String, Integer, BigInteger, ForeignKey, Enum as SQLEnum, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin, SoftDeleteMixin
from .enums import PhotoStatus



class Photo(Base, TimestampMixin, SoftDeleteMixin):
    """Photo model."""
    
    __tablename__ = 'photos'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    album_id = Column(UUID(as_uuid=True), ForeignKey('albums.id', ondelete='CASCADE'), nullable=False, index=True)
    uploader_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    
    # Storage info
    s3_key = Column(String(512), nullable=False, unique=True, index=True)
    s3_bucket = Column(String(255), nullable=False)
    
    # File metadata
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False, default='image/jpeg')
    filesize = Column(BigInteger, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    
    # Processing status
    status = Column(SQLEnum(PhotoStatus), default=PhotoStatus.uploaded, nullable=False, index=True)
    processing_error = Column(Text, nullable=True)
    
    # EXIF and metadata
    exif = Column(String, nullable=True)  # JSONB
    taken_at = Column(String, nullable=True)  # Extracted from EXIF
    camera_model = Column(String(255), nullable=True)
    
    # Thumbnails
    thumbnail_small_url = Column(String(512), nullable=True)
    thumbnail_medium_url = Column(String(512), nullable=True)
    thumbnail_large_url = Column(String(512), nullable=True)
    
    # Watermark
    watermarked_url = Column(String(512), nullable=True)
    
    # Additional metadata
    extra_data = Column(String, nullable=True)  # JSONB
    
    # Relationships
    album = relationship('Album', back_populates='photos')
    uploader = relationship('User', back_populates='uploaded_photos', foreign_keys=[uploader_id])
    faces = relationship('Face', back_populates='photo', cascade='all, delete-orphan')
    
    def __repr__(self) -> str:
        return f'<Photo(id={self.id}, filename={self.filename}, status={self.status})>'

