"""Download tracking model."""
from sqlalchemy import Column, String, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin
from db.base import Base


class Download(Base, TimestampMixin):
    """Track photo downloads."""
    
    __tablename__ = 'downloads'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    photo_id = Column(UUID(as_uuid=True), ForeignKey('photos.id', ondelete='CASCADE'), nullable=False, index=True)
    payment_id = Column(UUID(as_uuid=True), ForeignKey('payments.id', ondelete='SET NULL'), nullable=True)
    
    # Download details
    download_type = Column(String(50), nullable=False)  # 'free', 'paid', 'watermarked', 'hd'
    is_hd = Column(Boolean, default=False, nullable=False)
    
    # Request details
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(512), nullable=True)
    
    # Additional metadata
    extra_data = Column(String, nullable=True)  # JSONB
    
    # Relationships
    user = relationship('User')
    photo = relationship('Photo')
    payment = relationship('Payment')
    
    def __repr__(self) -> str:
        return f'<Download(id={self.id}, photo_id={self.photo_id})>'

