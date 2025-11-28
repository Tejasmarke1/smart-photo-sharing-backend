"""Login history model."""
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin


class LoginHistory(Base, TimestampMixin):
    """Track login attempts."""
    
    __tablename__ = "login_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(20), nullable=True, index=True)
    login_method = Column(String(50), nullable=False)
    is_successful = Column(Boolean, nullable=False)
    failure_reason = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(String(512), nullable=True)
    device_name = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    
    user = relationship("User", backref="login_history")