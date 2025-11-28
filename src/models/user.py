"""User model."""
from sqlalchemy import Column, String, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin, SoftDeleteMixin
from .enums import UserRole
from src.db.base import Base


class User(Base, TimestampMixin, SoftDeleteMixin):
    """User model for photographers, editors, guests, and admins."""
    
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(20), unique=True, nullable=True, index=True)
    hashed_password = Column(String(255), nullable=False)   
    role = Column(SQLEnum(UserRole), default=UserRole.guest, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    profile_picture_url = Column(String(512), nullable=True)
    
    # Optional metadata
    extra_data = Column(String, nullable=True)  # JSONB stored as string
    
    # Relationships
    albums = relationship('Album', back_populates='photographer', foreign_keys='Album.photographer_id')
    uploaded_photos = relationship('Photo', back_populates='uploader', foreign_keys='Photo.uploader_id')
    payments = relationship('Payment', back_populates='user')
    subscriptions = relationship('Subscription', back_populates='user')
    audit_logs = relationship('AuditLog', back_populates='actor', foreign_keys='AuditLog.actor_id')
    
    def __repr__(self) -> str:
        return f'<User(id={self.id}, email={self.email}, role={self.role})>'

