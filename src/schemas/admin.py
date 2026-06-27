"""Admin schemas for Pydantic v2."""
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from uuid import UUID
from typing import List, Optional, Any
from src.models.enums import UserRole, AuditAction


class AdminUserResponse(BaseModel):
    """User profile response for admin view."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    email: str
    phone: Optional[str] = None
    role: UserRole
    is_active: bool
    is_verified: bool
    profile_picture_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class UserListResponse(BaseModel):
    """Paginated list of users."""
    items: List[AdminUserResponse]
    total: int
    page: int
    size: int
    pages: int


class UserBanRequest(BaseModel):
    """Request schema to ban/unban a user."""
    is_active: bool = Field(..., description="Set False to ban, True to unban")
    reason: Optional[str] = Field(None, description="Reason for ban/unban action")


class PlatformStatsResponse(BaseModel):
    """Overall platform statistics response."""
    total_users: int
    total_photographers: int
    total_photos: int
    total_albums: int
    total_revenue_cents: int
    total_storage_bytes: int
    active_subscriptions: int


class AdminAuditLogResponse(BaseModel):
    """Audit log entry response."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    actor_id: Optional[UUID] = None
    actor_name: Optional[str] = None
    action: AuditAction
    target_type: str
    target_id: Optional[UUID] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[str] = None
    created_at: datetime


class AuditLogListResponse(BaseModel):
    """Paginated list of audit logs."""
    items: List[AdminAuditLogResponse]
    total: int
    page: int
    size: int
    pages: int
