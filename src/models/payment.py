"""Payment model."""
from sqlalchemy import Column, String, BigInteger, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin
from .enums import PaymentStatus



class Payment(Base, TimestampMixin):
    """Payment transaction model."""
    
    __tablename__ = 'payments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    album_id = Column(UUID(as_uuid=True), ForeignKey('albums.id', ondelete='SET NULL'), nullable=True, index=True)
    
    # Razorpay details
    razorpay_order_id = Column(String(255), unique=True, nullable=False, index=True)
    razorpay_payment_id = Column(String(255), unique=True, nullable=True, index=True)
    razorpay_signature = Column(String(255), nullable=True)
    
    # Payment details
    amount_cents = Column(BigInteger, nullable=False)  # Store in smallest currency unit
    currency = Column(String(3), default='INR', nullable=False)
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.pending, nullable=False, index=True)
    
    # Payment type
    payment_type = Column(String(50), nullable=False)  # 'download', 'subscription', 'print'
    
    # Reference details
    reference_type = Column(String(50), nullable=True)  # 'photo', 'album', 'subscription'
    reference_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Additional metadata
    extra_data = Column(String, nullable=True)  # JSONB
    
    # Error details
    error_code = Column(String(50), nullable=True)
    error_message = Column(String, nullable=True)
    
    # Relationships
    user = relationship('User', back_populates='payments')
    album = relationship('Album', back_populates='payments')
    
    def __repr__(self) -> str:
        return f'<Payment(id={self.id}, order_id={self.razorpay_order_id}, status={self.status})>'

