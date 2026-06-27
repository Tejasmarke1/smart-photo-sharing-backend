"""Payment and subscription endpoints."""
import json
import logging
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from src.db.base import get_db
from src.api.deps import get_current_user
from src.models.user import User
from src.models.plan import Plan
from src.models.payment import Payment
from src.models.subscription import Subscription
from src.models.enums import UserRole, PaymentStatus, SubscriptionStatus, PlanAudiance, BillingCycle
from src.schemas.payment import (
    PlanResponse,
    PaymentCreateRequest,
    PaymentCreateResponse,
    PaymentVerifyRequest,
    PaymentResponse,
    SubscriptionCreateRequest,
    SubscriptionResponse
)
from src.services.payments.razorpay_service import RazorpayService

logger = logging.getLogger(__name__)

router = APIRouter()
razorpay_service = RazorpayService()


@router.get("/plans", response_model=List[PlanResponse], summary="List active plans")
async def list_plans(
    role: Optional[PlanAudiance] = None,
    db: Session = Depends(get_db)
):
    """
    List all active subscription/pricing plans.
    Can be filtered by role (audience).
    """
    query = db.query(Plan).filter(Plan.is_active == True)
    if role:
        query = query.filter(Plan.role == role)
    
    plans = query.order_by(Plan.sort_order.asc()).all()
    
    # If no plans exist (e.g. fresh DB), seed some default plans for testing/dev
    if not plans and db.query(Plan).count() == 0:
        logger.info("No plans found in DB. Seeding default plans for dev.")
        default_plans = [
            Plan(
                id=uuid4(),
                code="free_tier",
                name="Free Tier",
                role=PlanAudiance.user,
                storage_limit_bytes=5 * 1024 * 1024 * 1024, # 5 GB
                price_cents=0,
                currency="INR",
                billing_cycle=BillingCycle.monthly,
                is_active=True,
                sort_order=1
            ),
            Plan(
                id=uuid4(),
                code="pro_photographer",
                name="Pro Photographer",
                role=PlanAudiance.photographer,
                storage_limit_bytes=100 * 1024 * 1024 * 1024, # 100 GB
                price_cents=99900, # Rs. 999
                currency="INR",
                billing_cycle=BillingCycle.monthly,
                is_active=True,
                sort_order=2
            ),
            Plan(
                id=uuid4(),
                code="pro_user",
                name="Pro Guest/User",
                role=PlanAudiance.user,
                storage_limit_bytes=20 * 1024 * 1024 * 1024, # 20 GB
                price_cents=19900, # Rs. 199
                currency="INR",
                billing_cycle=BillingCycle.monthly,
                is_active=True,
                sort_order=3
            )
        ]
        db.add_all(default_plans)
        db.commit()
        plans = db.query(Plan).filter(Plan.is_active == True).order_by(Plan.sort_order.asc()).all()
        
    return plans


@router.post("/payments/create-order", response_model=PaymentCreateResponse, summary="Create a payment order")
async def create_payment_order(
    payload: PaymentCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a payment record and initiate a Razorpay order.
    """
    # Use database transaction
    try:
        # Check reference if payment is for subscription
        if payload.payment_type == "subscription":
            if not payload.reference_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Plan ID (reference_id) is required for subscription payments"
                )
            plan = db.query(Plan).filter(Plan.id == payload.reference_id).first()
            if not plan:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Subscription plan not found"
                )
            payload.amount_cents = plan.price_cents
            payload.currency = plan.currency
            
        receipt_id = f"rcpt_{uuid4().hex[:20]}"
        
        # Call Razorpay to create order
        razorpay_order = razorpay_service.create_order(
            amount_cents=payload.amount_cents,
            currency=payload.currency,
            receipt=receipt_id
        )
        
        # Save payment transaction to DB
        payment = Payment(
            id=uuid4(),
            user_id=current_user.id,
            album_id=payload.album_id,
            razorpay_order_id=razorpay_order["id"],
            amount_cents=payload.amount_cents,
            currency=payload.currency,
            status=PaymentStatus.pending,
            payment_type=payload.payment_type,
            reference_type=payload.reference_type,
            reference_id=payload.reference_id
        )
        
        db.add(payment)
        db.commit()
        db.refresh(payment)
        
        return payment
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create payment order: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate payment: {str(e)}"
        )


@router.post("/payments/verify", response_model=PaymentResponse, summary="Verify payment signature")
async def verify_payment(
    payload: PaymentVerifyRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Verify payment signature returned by Razorpay frontend SDK.
    On success, updates transaction status and activates plan/downloads.
    """
    payment = db.query(Payment).filter(Payment.razorpay_order_id == payload.razorpay_order_id).first()
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment transaction not found"
        )
        
    # Verify signature
    is_valid = razorpay_service.verify_payment_signature(
        razorpay_order_id=payload.razorpay_order_id,
        razorpay_payment_id=payload.razorpay_payment_id,
        razorpay_signature=payload.razorpay_signature
    )
    
    if not is_valid:
        payment.status = PaymentStatus.failed
        payment.error_code = "BAD_SIGNATURE"
        payment.error_message = "Razorpay payment signature verification failed"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid payment signature"
        )
        
    try:
        # Update payment status
        payment.status = PaymentStatus.completed
        payment.razorpay_payment_id = payload.razorpay_payment_id
        payment.razorpay_signature = payload.razorpay_signature
        
        # If subscription, activate it
        if payment.payment_type == "subscription":
            plan = db.query(Plan).filter(Plan.id == payment.reference_id).first()
            if plan:
                # Cancel existing active subscriptions
                active_subs = db.query(Subscription).filter(
                    Subscription.user_id == current_user.id,
                    Subscription.status == SubscriptionStatus.active
                ).all()
                for sub in active_subs:
                    sub.status = SubscriptionStatus.cancelled
                    sub.cancelled_at = datetime.utcnow()
                
                # Determine end date
                days = 30 if plan.billing_cycle == BillingCycle.monthly else 365
                end_date = datetime.utcnow() + timedelta(days=days)
                
                new_sub = Subscription(
                    id=uuid4(),
                    user_id=current_user.id,
                    plan_id=plan.id,
                    status=SubscriptionStatus.active,
                    razorpay_plan_id=plan.razorpay_plan_id,
                    start_date=datetime.utcnow(),
                    end_date=end_date,
                    is_trial=False,
                    auto_renew=True
                )
                db.add(new_sub)
                
                # Update user role/storage limits
                # (Assuming User model is updated with new role if needed)
                current_user.role = UserRole.photographer if plan.role == PlanAudiance.photographer else UserRole.guest
                
        db.commit()
        db.refresh(payment)
        return payment
    except Exception as e:
        db.rollback()
        logger.error(f"Error completing payment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update database on successful payment"
        )


@router.get("/payments/history", response_model=List[PaymentResponse], summary="Get payment history")
async def get_payment_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get payment transaction history for current user.
    """
    payments = db.query(Payment).filter(Payment.user_id == current_user.id).order_by(Payment.created_at.desc()).all()
    return payments


@router.post("/subscriptions/create", response_model=SubscriptionResponse, summary="Subscribe to a plan")
async def create_subscription(
    payload: SubscriptionCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Subscribe to a plan directly (handles 100% free plans).
    Paid plans should go through /payments/create-order -> verify flow.
    """
    plan = db.query(Plan).filter(Plan.id == payload.plan_id, Plan.is_active == True).first()
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plan not found or inactive"
        )
        
    if plan.price_cents > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Paid plans must go through checkout flow"
        )
        
    try:
        # Cancel current active subscriptions
        active_subs = db.query(Subscription).filter(
            Subscription.user_id == current_user.id,
            Subscription.status == SubscriptionStatus.active
        ).all()
        for sub in active_subs:
            sub.status = SubscriptionStatus.cancelled
            sub.cancelled_at = datetime.utcnow()
            
        days = 30 if plan.billing_cycle == BillingCycle.monthly else 365
        end_date = datetime.utcnow() + timedelta(days=days)
        
        subscription = Subscription(
            id=uuid4(),
            user_id=current_user.id,
            plan_id=plan.id,
            status=SubscriptionStatus.active,
            start_date=datetime.utcnow(),
            end_date=end_date,
            is_trial=False,
            auto_renew=True
        )
        
        db.add(subscription)
        current_user.role = UserRole.photographer if plan.role == PlanAudiance.photographer else UserRole.guest
        db.commit()
        db.refresh(subscription)
        return subscription
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Subscription activation failed: {str(e)}"
        )


@router.post("/subscriptions/cancel", response_model=SubscriptionResponse, summary="Cancel subscription")
async def cancel_subscription(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Cancel active subscription of current user.
    """
    subscription = db.query(Subscription).filter(
        Subscription.user_id == current_user.id,
        Subscription.status == SubscriptionStatus.active
    ).first()
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )
        
    try:
        subscription.status = SubscriptionStatus.cancelled
        subscription.cancelled_at = datetime.utcnow()
        subscription.auto_renew = False
        db.commit()
        db.refresh(subscription)
        return subscription
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to cancel subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription"
        )


@router.get("/subscriptions/current", response_model=SubscriptionResponse, summary="Get current active subscription")
async def get_current_subscription(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user's current active subscription details.
    """
    subscription = db.query(Subscription).filter(
        Subscription.user_id == current_user.id,
        Subscription.status == SubscriptionStatus.active
    ).first()
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found for user"
        )
        
    return subscription


@router.post("/payments/webhook", summary="Razorpay Webhook handler")
async def razorpay_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Handles background status updates from Razorpay Webhooks.
    """
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature")
    
    if not signature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook signature header missing"
        )
        
    is_valid = razorpay_service.verify_webhook_signature(body, signature)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook signature verification failed"
        )
        
    try:
        event = json.loads(body.decode('utf-8'))
        event_type = event.get("event")
        logger.info(f"Received Razorpay webhook event: {event_type}")
        
        # Handle payment captured event
        if event_type == "payment.captured":
            payment_entity = event["payload"]["payment"]["entity"]
            order_id = payment_entity.get("order_id")
            payment_id = payment_entity.get("id")
            
            payment = db.query(Payment).filter(Payment.razorpay_order_id == order_id).first()
            if payment and payment.status != PaymentStatus.completed:
                payment.status = PaymentStatus.completed
                payment.razorpay_payment_id = payment_id
                
                # If subscription, activate it
                if payment.payment_type == "subscription":
                    user = db.query(User).filter(User.id == payment.user_id).first()
                    plan = db.query(Plan).filter(Plan.id == payment.reference_id).first()
                    if user and plan:
                        # Cancel existing
                        active_subs = db.query(Subscription).filter(
                            Subscription.user_id == user.id,
                            Subscription.status == SubscriptionStatus.active
                        ).all()
                        for sub in active_subs:
                            sub.status = SubscriptionStatus.cancelled
                            sub.cancelled_at = datetime.utcnow()
                            
                        days = 30 if plan.billing_cycle == BillingCycle.monthly else 365
                        end_date = datetime.utcnow() + timedelta(days=days)
                        
                        new_sub = Subscription(
                            id=uuid4(),
                            user_id=user.id,
                            plan_id=plan.id,
                            status=SubscriptionStatus.active,
                            razorpay_plan_id=plan.razorpay_plan_id,
                            start_date=datetime.utcnow(),
                            end_date=end_date,
                            is_trial=False,
                            auto_renew=True
                        )
                        db.add(new_sub)
                        user.role = UserRole.photographer if plan.role == PlanAudiance.photographer else UserRole.guest
                
                db.commit()
                logger.info(f"Payment marked completed via webhook: {order_id}")
                
        elif event_type == "payment.failed":
            payment_entity = event["payload"]["payment"]["entity"]
            order_id = payment_entity.get("order_id")
            error_code = payment_entity.get("error_code")
            error_message = payment_entity.get("error_description")
            
            payment = db.query(Payment).filter(Payment.razorpay_order_id == order_id).first()
            if payment and payment.status != PaymentStatus.completed:
                payment.status = PaymentStatus.failed
                payment.error_code = error_code
                payment.error_message = error_message
                db.commit()
                logger.info(f"Payment marked failed via webhook: {order_id}")
                
        return {"status": "ok"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error handling Razorpay webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook payload"
        )
