"""Notification schemas for Pydantic v2."""
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from uuid import UUID
from typing import List, Optional, Dict, Any
from enum import Enum


class DeviceType(str, Enum):
    android = "android"
    ios = "ios"
    web = "web"


class DeviceTokenRegisterRequest(BaseModel):
    """Register a device token for push notifications."""
    token: str = Field(..., min_length=10, max_length=512, description="FCM device token")
    device_type: DeviceType = Field(..., description="Device platform")
    device_name: Optional[str] = Field(None, max_length=255, description="Human-readable device name")


class DeviceTokenResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    token: str
    device_type: str
    device_name: Optional[str] = None
    is_active: bool
    created_at: datetime


class NotificationTarget(str, Enum):
    all = "all"
    photographers = "photographers"
    users = "users"


class SendNotificationRequest(BaseModel):
    """Request to send a push notification."""
    title: str = Field(..., min_length=1, max_length=255, description="Notification title")
    body: str = Field(..., min_length=1, max_length=2000, description="Notification body text")
    image_url: Optional[str] = Field(None, max_length=1024, description="Optional image URL for rich notification (marketing)")
    target: str = Field(..., description="Target audience: 'all', 'photographers', 'users', or a specific user UUID")
    data: Optional[Dict[str, str]] = Field(None, description="Optional data payload for deep links, e.g. {'screen': '/album/abc123'}")


class NotificationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    sender_id: Optional[UUID] = None
    title: str
    body: str
    image_url: Optional[str] = None
    target: str
    data: Optional[str] = None
    sent_count: int
    fail_count: int
    created_at: datetime


class NotificationListResponse(BaseModel):
    items: List[NotificationResponse]
    total: int
    page: int
    size: int
    pages: int
