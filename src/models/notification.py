"""Notification model for tracking sent push notifications."""
from sqlalchemy import Column, String, Text, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin


class Notification(Base, TimestampMixin):
    """Track sent push notifications."""
    __tablename__ = 'notifications'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    sender_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    title = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)
    image_url = Column(String(1024), nullable=True)
    target = Column(String(255), nullable=False)  # 'all', 'photographers', 'users', or UUID string
    data = Column(Text, nullable=True)  # JSON string for deep links
    sent_count = Column(Integer, default=0, nullable=False)
    fail_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    sender = relationship('User')
    
    def __repr__(self) -> str:
        return f'<Notification(id={self.id}, title={self.title}, target={self.target})>'
