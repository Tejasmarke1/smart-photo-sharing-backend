"""Subscription model."""
from sqlalchemy import Column, String, DateTime, ForeignKey, Enum as SQLEnum, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin
from .enums import SubscriptionPlan, SubscriptionStatus
from db.base import Base


class Subscription(Base, TimestampMixin):
    """User subscription model."""
    
    __tablename__ = 'subscriptions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Subscription details
    plan = Column(SQLEnum(SubscriptionPlan), nullable=False)
    status = Column(SQLEnum(SubscriptionStatus), default=SubscriptionStatus.active, nullable=False, index=True)
    
    # Razorpay subscription details
    razorpay_subscription_id = Column(String(255), unique=True, nullable=True, index=True)
    razorpay_plan_id = Column(String(255), nullable=True)
    
    # Dates
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    trial_end_date = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    # Billing
    is_trial = Column(Boolean, default=False, nullable=False)
    auto_renew = Column(Boolean, default=True, nullable=False)
    
    # Additional metadata
    extra_data = Column(String, nullable=True)  # JSONB
    
    # Relationships
    user = relationship('User', back_populates='subscriptions')
    
    def __repr__(self) -> str:
        return f'<Subscription(id={self.id}, user_id={self.user_id}, plan={self.plan})>'

