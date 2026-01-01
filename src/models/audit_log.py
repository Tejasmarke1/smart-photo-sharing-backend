"""Audit log model."""
from sqlalchemy import Column, String, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin
from .enums import AuditAction



class AuditLog(Base, TimestampMixin):
    """Audit log for tracking user actions."""
    
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    actor_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    
    # Action details
    action = Column(SQLEnum(AuditAction,name="auditaction"), nullable=False, index=True)
    target_type = Column(String(50), nullable=False, index=True)  # 'album', 'photo', 'user', etc.
    target_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Request details
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(512), nullable=True)
    
    # Additional details
    details = Column(String, nullable=True)  # JSONB
    
    # Relationships
    actor = relationship('User', back_populates='audit_logs', foreign_keys=[actor_id])
    
    def __repr__(self) -> str:
        return f'<AuditLog(id={self.id}, action={self.action}, target_type={self.target_type})>'

