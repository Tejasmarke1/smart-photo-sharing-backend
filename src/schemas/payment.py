"""Payment and Subscription schemas."""
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from uuid import UUID
from typing import Optional, Any
from src.models.enums import PaymentStatus, SubscriptionStatus, BillingCycle, PlanAudiance


class PlanResponse(BaseModel):
    """Schema for subscription/pricing plans."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    code: str
    name: str
    role: PlanAudiance
    storage_limit_bytes: int
    price_cents: int
    currency: str
    billing_cycle: BillingCycle
    razorpay_plan_id: Optional[str] = None
    is_active: bool
    sort_order: int
    created_at: datetime
    updated_at: datetime


class PaymentCreateRequest(BaseModel):
    """Request to initiate a payment order (Razorpay)."""
    amount_cents: int = Field(..., gt=0, description="Amount in cents/smallest currency unit")
    currency: str = Field("INR", min_length=3, max_length=3)
    payment_type: str = Field(..., description="'download', 'subscription', or 'print'")
    album_id: Optional[UUID] = None
    reference_type: Optional[str] = Field(None, description="'photo', 'album', or 'subscription'")
    reference_id: Optional[UUID] = None


class PaymentCreateResponse(BaseModel):
    """Response after creating a payment transaction order."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    razorpay_order_id: str
    amount_cents: int
    currency: str
    status: PaymentStatus
    payment_type: str
    album_id: Optional[UUID] = None
    reference_type: Optional[str] = None
    reference_id: Optional[UUID] = None


class PaymentVerifyRequest(BaseModel):
    """Request to verify payment signature after frontend checkout."""
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class PaymentResponse(BaseModel):
    """Details of a payment transaction."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: Optional[UUID] = None
    album_id: Optional[UUID] = None
    razorpay_order_id: str
    razorpay_payment_id: Optional[str] = None
    amount_cents: int
    currency: str
    status: PaymentStatus
    payment_type: str
    reference_type: Optional[str] = None
    reference_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime


class SubscriptionCreateRequest(BaseModel):
    """Request to create a new subscription."""
    plan_id: UUID


class SubscriptionResponse(BaseModel):
    """Subscription details."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    plan_id: UUID
    status: SubscriptionStatus
    razorpay_subscription_id: Optional[str] = None
    razorpay_plan_id: Optional[str] = None
    start_date: datetime
    end_date: Optional[datetime] = None
    trial_end_date: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    is_trial: bool
    auto_renew: bool
    created_at: datetime
    updated_at: datetime
