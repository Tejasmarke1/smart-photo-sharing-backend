"""
Admin Dashboard Integration Tests
=================================
Tests for user management, banning/unbanning, platform stats, audit logs,
and role-based access control (RBAC).
"""

import pytest
from uuid import uuid4, UUID
from datetime import datetime
from fastapi import status

from src.models.user import User
from src.models.photo import Photo
from src.models.album import Album
from src.models.payment import Payment
from src.models.subscription import Subscription
from src.models.audit_log import AuditLog
from src.models.enums import UserRole, PaymentStatus, SubscriptionStatus, AuditAction
from src.api.deps import get_current_user


@pytest.fixture
def admin_user(db):
    """Create a test admin user in DB."""
    admin = User(
        id=uuid4(),
        name="Platform Admin",
        email="admin@example.com",
        phone="+1112223333",
        hashed_password="hashed_password",
        role=UserRole.admin,
        is_active=True,
        is_verified=True
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return admin


@pytest.fixture
def guest_user(db):
    """Create a test guest/regular user in DB."""
    user = User(
        id=uuid4(),
        name="Guest User",
        email="guest@example.com",
        phone="+9998887777",
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
def admin_client(client, admin_user):
    """Client authenticated as Admin."""
    def override_get_current_user():
        return admin_user
    
    client.app.dependency_overrides[get_current_user] = override_get_current_user
    yield client
    client.app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def guest_client(client, guest_user):
    """Client authenticated as Guest/Non-Admin."""
    def override_get_current_user():
        return guest_user
    
    client.app.dependency_overrides[get_current_user] = override_get_current_user
    yield client
    client.app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.integration
def test_list_users_forbidden_for_guest(guest_client):
    """Test that non-admin user cannot list users (403 Forbidden)."""
    response = guest_client.get("/api/v1/admin/users")
    assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.integration
def test_list_users_success_as_admin(admin_client, db, guest_user):
    """Test listing users as admin."""
    response = admin_client.get("/api/v1/admin/users")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "items" in data
    assert data["total"] >= 2  # Admin + Guest user
    
    names = [user["name"] for user in data["items"]]
    assert "Guest User" in names
    assert "Platform Admin" in names


@pytest.mark.integration
def test_list_users_filters(admin_client, db, guest_user):
    """Test listing users with filters (role and search)."""
    # Create a photographer
    photographer = User(
        id=uuid4(),
        name="John Photographer",
        email="john@example.com",
        phone="+5554443333",
        hashed_password="password",
        role=UserRole.photographer,
        is_active=True,
        is_verified=True
    )
    db.add(photographer)
    db.commit()

    # Filter by role
    response = admin_client.get("/api/v1/admin/users?role=photographer")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["name"] == "John Photographer"

    # Filter by search string
    response = admin_client.get("/api/v1/admin/users?search=john")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["email"] == "john@example.com"


@pytest.mark.integration
def test_ban_user_success(admin_client, db, guest_user):
    """Test banning a user successfully as admin."""
    payload = {
        "is_active": False,
        "reason": "Violation of terms of service"
    }
    response = admin_client.put(f"/api/v1/admin/users/{guest_user.id}/ban", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["is_active"] is False

    # Check DB
    db.refresh(guest_user)
    assert guest_user.is_active is False

    # Check Audit Log was created
    audit_log = db.query(AuditLog).filter(
        AuditLog.target_id == guest_user.id,
        AuditLog.action == AuditAction.update
    ).first()
    assert audit_log is not None
    assert "Violation of terms" in audit_log.details


@pytest.mark.integration
def test_ban_self_forbidden(admin_client, admin_user):
    """Test that admin cannot ban themselves."""
    payload = {
        "is_active": False,
        "reason": "Self harm"
    }
    response = admin_client.put(f"/api/v1/admin/users/{admin_user.id}/ban", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "You cannot ban yourself" in response.json()["detail"]


@pytest.mark.integration
def test_platform_stats(admin_client, db, admin_user, guest_user):
    """Test retrieving platform dashboard stats."""
    # Seed some mock data
    album = Album(
        id=uuid4(),
        photographer_id=admin_user.id,
        title="Test Album",
        sharing_code="ABCDEF",
        created_at=datetime.utcnow()
    )
    db.add(album)
    db.commit()

    photo = Photo(
        id=uuid4(),
        album_id=album.id,
        s3_key="key",
        s3_bucket="my-bucket",
        filename="test.jpg",
        status="done",
        filesize=2048,
        created_at=datetime.utcnow()
    )
    db.add(photo)
    
    payment = Payment(
        id=uuid4(),
        user_id=guest_user.id,
        razorpay_order_id="order_1",
        amount_cents=50000,
        currency="INR",
        status=PaymentStatus.completed,
        payment_type="subscription",
        reference_type="plan",
        reference_id=uuid4()
    )
    db.add(payment)

    sub = Subscription(
        id=uuid4(),
        user_id=guest_user.id,
        plan_id=uuid4(),
        status=SubscriptionStatus.active,
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow()
    )
    db.add(sub)
    db.commit()

    response = admin_client.get("/api/v1/admin/stats")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total_users"] == 2
    assert data["total_photos"] == 1
    assert data["total_albums"] == 1
    assert data["total_revenue_cents"] == 50000
    assert data["total_storage_bytes"] == 2048
    assert data["active_subscriptions"] == 1


@pytest.mark.integration
def test_get_audit_logs(admin_client, db, admin_user, guest_user):
    """Test retrieving platform audit logs."""
    audit_log = AuditLog(
        id=uuid4(),
        actor_id=admin_user.id,
        action=AuditAction.update,
        target_type="user",
        target_id=guest_user.id,
        details="Banned guest user"
    )
    db.add(audit_log)
    db.commit()

    response = admin_client.get("/api/v1/admin/audit-logs")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "items" in data
    assert data["total"] == 1
    assert data["items"][0]["actor_name"] == "Platform Admin"
    assert data["items"][0]["details"] == "Banned guest user"
