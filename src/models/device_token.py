"""Device token model for FCM push notifications."""
from sqlalchemy import Column, String, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin


class DeviceToken(Base, TimestampMixin):
    """Store FCM device tokens for push notifications."""
    __tablename__ = 'device_tokens'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    token = Column(String(512), nullable=False, unique=True)
    device_type = Column(String(20), nullable=False)  # 'android', 'ios', 'web'
    device_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship('User')
    
    def __repr__(self) -> str:
        return f'<DeviceToken(id={self.id}, user_id={self.user_id}, type={self.device_type})>'
