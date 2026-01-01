from sqlalchemy import Column, Integer, String, Float, Boolean, BigInteger
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SQLEnum
from src.models.enums import UserRole , BillingCycle

import uuid
from src.db.base import Base
from .base import TimestampMixin
class Plan(Base, TimestampMixin):
    """plans model"""
    __tablename__ = 'plans'
    
    id = Column(UUID(as_uuid=True),primary_key=True, default=uuid.uuid4, index=True)
    code = Column(String((50)),nullable=False, unique=True)
    name = Column(String((100)),nullable=False, unique=True)
    role = Column(SQLEnum(UserRole,name="userrole"), default=UserRole.guest, nullable=False,index=True)
    
    storage_limit_bytes = Column(BigInteger, nullable=False, default=5.0,index=True)  # Storage limit in GB
    price_cents= Column(BigInteger, nullable=False, default=0, index=True)  # Price in cents
    currency = Column(String(3), default='INR', nullable=False)
    billing_cycle = Column(SQLEnum(BillingCycle,name="billingcycle"),nullable=False)
    razorpay_plan_id = Column(String(100), nullable=True)  # Razorpay plan ID for integration
    is_active = Column(Boolean,nullable=False,default=True)
    sort_order = Column(Integer, nullable=False, default=0)
    

    def __repr__(self) -> str:
        return f'<Plan(id={self.id}, code={self.code}, role={self.role}, price_cents={self.price_cents})>'
    
    


