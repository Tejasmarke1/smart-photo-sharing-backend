from sqlalchemy import Column, Integer, String, Float, Boolean , DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SQLEnum
import uuid
from datetime import datetime

from src.db.base import Base
from .base import TimestampMixin


class StorageUsage(Base, TimestampMixin):
    """Storage usage model to track user storage consumption."""
    
    __tablename__ = 'storage_usages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Storage usage details
    used_bytes = Column(Float, nullable=False, default=0.0)  # Used storage in bytes
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='storage_usage')
    
    def __repr__(self) -> str:
        return f'<StorageUsage(id={self.id}, user_id={self.user_id}, used_bytes={self.used_bytes})>'