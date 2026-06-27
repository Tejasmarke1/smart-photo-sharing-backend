"""Admin dashboard endpoints."""
import logging
from math import ceil
from uuid import UUID
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.db.base import get_db
from src.api.deps import get_current_user, require_roles
from src.models.user import User
from src.models.photo import Photo
from src.models.album import Album
from src.models.payment import Payment
from src.models.subscription import Subscription
from src.models.audit_log import AuditLog
from src.models.device_token import DeviceToken
from src.models.notification import Notification
from src.models.enums import UserRole, PaymentStatus, SubscriptionStatus, AuditAction
from src.schemas.admin import (
    AdminUserResponse,
    UserListResponse,
    UserBanRequest,
    PlatformStatsResponse,
    AdminAuditLogResponse,
    AuditLogListResponse
)
from src.schemas.notification import (
    SendNotificationRequest,
    NotificationResponse,
    NotificationListResponse
)
from src.services.messaging.fcm_service import FCMService
import json

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/users", response_model=UserListResponse, summary="List users (paginated)")
@require_roles(["admin"])
async def list_users(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    role: Optional[UserRole] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all platform users with filtering and pagination.
    Protected to Admin role.
    """
    query = db.query(User)
    
    if role:
        query = query.filter(User.role == role)
        
    if search:
        query = query.filter(
            (User.name.ilike(f"%{search}%")) | 
            (User.email.ilike(f"%{search}%")) |
            (User.phone.ilike(f"%{search}%"))
        )
        
    total = query.count()
    pages = ceil(total / size) if total > 0 else 0
    offset = (page - 1) * size
    
    users = query.order_by(User.created_at.desc()).offset(offset).limit(size).all()
    
    return {
        "items": users,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages
    }


@router.put("/users/{user_id}/ban", response_model=AdminUserResponse, summary="Ban or unban user")
@require_roles(["admin"])
async def ban_user(
    user_id: UUID,
    payload: UserBanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Ban a user by setting is_active = False or unban by setting is_active = True.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
        
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot ban yourself"
        )
        
    try:
        user.is_active = payload.is_active
        
        # Log this admin action to Audit Log
        action_detail = f"User {user.email} ban status updated to active={payload.is_active}. Reason: {payload.reason or 'None'}"
        audit_log = AuditLog(
            actor_id=current_user.id,
            action=AuditAction.update,
            target_type="user",
            target_id=user.id,
            details=action_detail
        )
        db.add(audit_log)
        
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update ban status for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user status"
        )


@router.get("/stats", response_model=PlatformStatsResponse, summary="Get platform statistics")
@require_roles(["admin"])
async def get_platform_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get aggregated usage statistics for the smart photo sharing platform.
    """
    try:
        total_users = db.query(User).count()
        total_photographers = db.query(User).filter(User.role == UserRole.photographer).count()
        total_photos = db.query(Photo).count()
        total_albums = db.query(Album).count()
        
        # Total storage size
        total_storage = db.query(func.sum(Photo.filesize)).scalar() or 0
        
        # Total revenue from payments
        total_rev = db.query(func.sum(Payment.amount_cents)).filter(
            Payment.status == PaymentStatus.completed
        ).scalar() or 0
        
        # Active subscriptions
        active_subs = db.query(Subscription).filter(
            Subscription.status == SubscriptionStatus.active
        ).count()
        
        return {
            "total_users": total_users,
            "total_photographers": total_photographers,
            "total_photos": total_photos,
            "total_albums": total_albums,
            "total_revenue_cents": total_rev,
            "total_storage_bytes": total_storage,
            "active_subscriptions": active_subs
        }
    except Exception as e:
        logger.error(f"Failed to fetch platform stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics data"
        )


@router.get("/audit-logs", response_model=AuditLogListResponse, summary="Get platform audit logs")
@require_roles(["admin"])
async def get_audit_logs(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    action: Optional[AuditAction] = None,
    actor_id: Optional[UUID] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Fetch platform audit logs with pagination and filters.
    """
    query = db.query(AuditLog)
    
    if action:
        query = query.filter(AuditLog.action == action)
    if actor_id:
        query = query.filter(AuditLog.actor_id == actor_id)
        
    total = query.count()
    pages = ceil(total / size) if total > 0 else 0
    offset = (page - 1) * size
    
    logs = query.order_by(AuditLog.created_at.desc()).offset(offset).limit(size).all()
    
    # Map logs to serialize actor names
    mapped_logs = []
    for log in logs:
        actor_name = log.actor.name if log.actor else "System"
        mapped_logs.append(
            AdminAuditLogResponse(
                id=log.id,
                actor_id=log.actor_id,
                actor_name=actor_name,
                action=log.action,
                target_type=log.target_type,
                target_id=log.target_id,
                ip_address=log.ip_address,
                user_agent=log.user_agent,
                details=log.details,
                created_at=log.created_at
            )
        )
        
    return {
        "items": mapped_logs,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages
    }


@router.post("/notifications/send", response_model=NotificationResponse, summary="Send push notification")
@require_roles(["admin"])
async def send_notification(
    payload: SendNotificationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Send a push notification to users.
    
    Target options:
    - 'all': Send to all active users
    - 'photographers': Send to photographers only
    - 'users': Send to non-photographer users only  
    - UUID string: Send to a specific user
    """
    # Build query for target device tokens
    query = db.query(DeviceToken).filter(DeviceToken.is_active == True)
    
    if payload.target == 'all':
        pass  # No additional filter
    elif payload.target == 'photographers':
        query = query.join(User).filter(User.role == UserRole.photographer)
    elif payload.target == 'users':
        query = query.join(User).filter(User.role != UserRole.photographer, User.role != UserRole.admin)
    else:
        # Assume it's a specific user UUID
        try:
            target_user_id = UUID(payload.target)
            query = query.filter(DeviceToken.user_id == target_user_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid target: {payload.target}. Use 'all', 'photographers', 'users', or a valid UUID."
            )
    
    tokens = [dt.token for dt in query.all()]
    
    # Send via FCM
    fcm = FCMService()
    data_payload = {k: str(v) for k, v in payload.data.items()} if payload.data else None
    success, failures, failed_tokens = fcm.send_to_tokens(
        tokens=tokens,
        title=payload.title,
        body=payload.body,
        image_url=payload.image_url,
        data=data_payload,
    )
    
    # Deactivate failed tokens
    if failed_tokens:
        db.query(DeviceToken).filter(
            DeviceToken.token.in_(failed_tokens)
        ).update({DeviceToken.is_active: False}, synchronize_session=False)
    
    # Record notification
    notification = Notification(
        sender_id=current_user.id,
        title=payload.title,
        body=payload.body,
        image_url=payload.image_url,
        target=payload.target,
        data=json.dumps(payload.data) if payload.data else None,
        sent_count=success,
        fail_count=failures,
    )
    db.add(notification)
    
    # Audit log
    audit_log = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.create,
        target_type="notification",
        target_id=notification.id,
        details=f"Sent push notification '{payload.title}' to target={payload.target}. Delivered: {success}, Failed: {failures}"
    )
    db.add(audit_log)
    
    db.commit()
    db.refresh(notification)
    return notification


@router.get("/notifications", response_model=NotificationListResponse, summary="List sent notifications")
@require_roles(["admin"])
async def list_notifications(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all sent notifications with pagination and delivery stats."""
    query = db.query(Notification)
    total = query.count()
    pages = ceil(total / size) if total > 0 else 0
    offset = (page - 1) * size
    
    notifications = query.order_by(Notification.created_at.desc()).offset(offset).limit(size).all()
    
    return {
        "items": notifications,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages
    }


@router.get("/notifications/{notification_id}", response_model=NotificationResponse, summary="Get notification detail")
@require_roles(["admin"])
async def get_notification(
    notification_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get details of a specific sent notification."""
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )
    return notification
