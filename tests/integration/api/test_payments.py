"""
Payments Integration Tests
=========================
Tests for plans, payment checkout, signature verification, subscription management,
and Razorpay webhooks.
"""

import pytest
import json
from uuid import uuid4, UUID
from datetime import datetime, timedelta
from fastapi import status

from src.models.user import User
from src.models.plan import Plan
from src.models.payment import Payment
from src.models.subscription import Subscription
from src.models.enums import UserRole, PaymentStatus, SubscriptionStatus, PlanAudiance, BillingCycle
from src.api.deps import get_current_user


@pytest.fixture(autouse=True)
def mock_razorpay_service(mocker):
    """Mock RazorpayService methods to prevent external API calls."""
    from src.api.v1.endpoints.payments import razorpay_service
    
    # Mock create_order
    def mock_create_order(amount_cents, currency="INR", receipt=None):
        return {
            "id": f"order_dummy_{receipt or '123'}",
            "entity": "order",
            "amount": amount_cents,
            "amount_paid": 0,
            "amount_due": amount_cents,
            "currency": currency,
            "receipt": receipt,
            "status": "created",
            "attempts": 0,
            "notes": {},
            "created_at": 1600000000
        }
    
    mocker.patch.object(razorpay_service, 'create_order', side_effect=mock_create_order)
    mocker.patch.object(razorpay_service, 'verify_payment_signature', return_value=True)
    mocker.patch.object(razorpay_service, 'verify_webhook_signature', return_value=True)


@pytest.fixture
def test_user(db):
    """Create a test user in DB."""
    user = User(
        id=uuid4(),
        name="Test Payments User",
        email="payments_user@example.com",
        phone="+1234567890",
        hashed_password="hashed_password",
        role=UserRole.guest,
        is_active=True,
        is_verified=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def auth_client(client, test_user):
    """FastAPI test client with dependency override for authenticated user."""
    def override_get_current_user():
        return test_user
    
    client.app.dependency_overrides[get_current_user] = override_get_current_user
    yield client
    client.app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.integration
def test_list_plans(auth_client, db):
    """Test listing pricing/subscription plans."""
    response = auth_client.get("/api/v1/payments/plans")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) >= 3
    # Check seeded plan codes
    codes = [plan["code"] for plan in data]
    assert "free_tier" in codes
    assert "pro_photographer" in codes
    assert "pro_user" in codes


@pytest.mark.integration
def test_create_payment_order_subscription(auth_client, db):
    """Test creating a payment order for a plan subscription."""
    # Ensure plans are seeded
    auth_client.get("/api/v1/payments/plans")
    plan = db.query(Plan).filter(Plan.code == "pro_photographer").first()
    assert plan is not None

    payload = {
        "payment_type": "subscription",
        "reference_type": "plan",
        "reference_id": str(plan.id),
        "amount_cents": plan.price_cents,
        "currency": plan.currency
    }
    response = auth_client.post("/api/v1/payments/payments/create-order", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "razorpay_order_id" in data
    assert data["status"] == "pending"
    assert data["amount_cents"] == plan.price_cents

    # Verify database record
    db_payment = db.query(Payment).filter(Payment.razorpay_order_id == data["razorpay_order_id"]).first()
    assert db_payment is not None
    assert db_payment.status == PaymentStatus.pending
    assert db_payment.payment_type == "subscription"


@pytest.mark.integration
def test_verify_payment_success(auth_client, db, test_user):
    """Test successful payment signature verification."""
    # Seed plan
    auth_client.get("/api/v1/payments/plans")
    plan = db.query(Plan).filter(Plan.code == "pro_photographer").first()

    # Create a pending payment
    payment = Payment(
        id=uuid4(),
        user_id=test_user.id,
        razorpay_order_id="order_dummy_123",
        amount_cents=plan.price_cents,
        currency=plan.currency,
        status=PaymentStatus.pending,
        payment_type="subscription",
        reference_type="plan",
        reference_id=plan.id
    )
    db.add(payment)
    db.commit()

    payload = {
        "razorpay_order_id": "order_dummy_123",
        "razorpay_payment_id": "pay_dummy_abc",
        "razorpay_signature": "sig_dummy_xyz"
    }
    response = auth_client.post("/api/v1/payments/payments/verify", json=payload)
    assert response.status_code == status.HTTP_200_OK
    
    # Reload payment and check status
    db.refresh(payment)
    assert payment.status == PaymentStatus.completed
    assert payment.razorpay_payment_id == "pay_dummy_abc"

    # Verify subscription is created
    subscription = db.query(Subscription).filter(
        Subscription.user_id == test_user.id,
        Subscription.plan_id == plan.id
    ).first()
    assert subscription is not None
    assert subscription.status == SubscriptionStatus.active

    # Verify user role updated
    db.refresh(test_user)
    assert test_user.role == UserRole.photographer


@pytest.mark.integration
def test_get_payment_history(auth_client, db, test_user):
    """Test fetching payment transaction history."""
    payment = Payment(
        id=uuid4(),
        user_id=test_user.id,
        razorpay_order_id="order_hist_1",
        amount_cents=1000,
        currency="INR",
        status=PaymentStatus.completed,
        payment_type="download",
        reference_type="photo",
        reference_id=uuid4()
    )
    db.add(payment)
    db.commit()

    response = auth_client.get("/api/v1/payments/payments/history")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 1
    assert data[0]["razorpay_order_id"] == "order_hist_1"


@pytest.mark.integration
def test_create_subscription_free_tier(auth_client, db, test_user):
    """Test subscribing directly to a free plan."""
    auth_client.get("/api/v1/payments/plans")
    free_plan = db.query(Plan).filter(Plan.code == "free_tier").first()

    payload = {
        "plan_id": str(free_plan.id)
    }
    response = auth_client.post("/api/v1/payments/subscriptions/create", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "active"
    assert data["plan_id"] == str(free_plan.id)

    # Verify user role updated to guest/user role
    db.refresh(test_user)
    assert test_user.role == UserRole.guest


@pytest.mark.integration
def test_create_subscription_paid_fails(auth_client, db):
    """Test directly subscribing to a paid plan fails."""
    auth_client.get("/api/v1/payments/plans")
    paid_plan = db.query(Plan).filter(Plan.code == "pro_photographer").first()

    payload = {
        "plan_id": str(paid_plan.id)
    }
    response = auth_client.post("/api/v1/payments/subscriptions/create", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Paid plans must go through checkout flow" in response.json()["detail"]


@pytest.mark.integration
def test_cancel_subscription(auth_client, db, test_user):
    """Test cancelling an active subscription."""
    auth_client.get("/api/v1/payments/plans")
    plan = db.query(Plan).filter(Plan.code == "free_tier").first()

    # Create active subscription
    sub = Subscription(
        id=uuid4(),
        user_id=test_user.id,
        plan_id=plan.id,
        status=SubscriptionStatus.active,
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow() + timedelta(days=30),
        auto_renew=True
    )
    db.add(sub)
    db.commit()

    response = auth_client.post("/api/v1/payments/subscriptions/cancel")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "cancelled"
    assert data["auto_renew"] is False

    db.refresh(sub)
    assert sub.status == SubscriptionStatus.cancelled


@pytest.mark.integration
def test_get_current_subscription(auth_client, db, test_user):
    """Test fetching active subscription."""
    auth_client.get("/api/v1/payments/plans")
    plan = db.query(Plan).filter(Plan.code == "free_tier").first()

    sub = Subscription(
        id=uuid4(),
        user_id=test_user.id,
        plan_id=plan.id,
        status=SubscriptionStatus.active,
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow() + timedelta(days=30),
        auto_renew=True
    )
    db.add(sub)
    db.commit()

    response = auth_client.get("/api/v1/payments/subscriptions/current")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == str(sub.id)


@pytest.mark.integration
def test_razorpay_webhook_captured(auth_client, db, test_user):
    """Test Razorpay captured webhook updates transaction."""
    auth_client.get("/api/v1/payments/plans")
    plan = db.query(Plan).filter(Plan.code == "pro_photographer").first()

    payment = Payment(
        id=uuid4(),
        user_id=test_user.id,
        razorpay_order_id="order_webhook_captured",
        amount_cents=plan.price_cents,
        currency=plan.currency,
        status=PaymentStatus.pending,
        payment_type="subscription",
        reference_type="plan",
        reference_id=plan.id
    )
    db.add(payment)
    db.commit()

    webhook_payload = {
        "event": "payment.captured",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_webhook_123",
                    "order_id": "order_webhook_captured",
                    "amount": plan.price_cents,
                    "currency": "INR",
                    "status": "captured"
                }
            }
        }
    }

    # Set webhook signature header
    headers = {
        "X-Razorpay-Signature": "dummy_sig"
    }

    # Webhook endpoint is public (no auth needed)
    response = auth_client.post(
        "/api/v1/payments/payments/webhook", 
        json=webhook_payload, 
        headers=headers
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}

    db.refresh(payment)
    assert payment.status == PaymentStatus.completed
    assert payment.razorpay_payment_id == "pay_webhook_123"

    # Verify subscription is created
    sub = db.query(Subscription).filter(Subscription.user_id == test_user.id).first()
    assert sub is not None
    assert sub.status == SubscriptionStatus.active

    db.refresh(test_user)
    assert test_user.role == UserRole.photographer
