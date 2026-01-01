"""Enums for database models."""
import enum


class UserRole(str, enum.Enum):
    """User role types."""
    photographer = "photographer"
    editor = "editor"
    guest = "guest"
    admin = "admin"


class PhotoStatus(str, enum.Enum):
    """Photo processing status."""
    uploaded = "uploaded"
    processing = "processing"
    done = "done"
    failed = "failed"


class PaymentStatus(str, enum.Enum):
    """Payment status."""
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    refunded = "refunded"


class SubscriptionPlan(str, enum.Enum):
    """Subscription plan types."""
    free = "free"
    basic = "basic"
    pro = "pro"
    enterprise = "enterprise"
    
    
class BillingCycle(str, enum.Enum):
    """Billing cycle types."""
    monthly = "monthly"
    yearly = "yearly"


class SubscriptionStatus(str, enum.Enum):
    """Subscription status."""
    active = "active"
    cancelled = "cancelled"
    expired = "expired"
    suspended = "suspended"


class AuditAction(str, enum.Enum):
    """Audit log action types."""
    create = "create"
    read = "read"
    update = "update"
    delete = "delete"
    login = "login"
    logout = "logout"
    download = "download"
    share = "share"
    
    
class FaceQuality(str, enum.Enum):
    """Face detection quality."""
    high = 'high'
    medium = 'medium'
    low = 'low'
    unknown = 'unknown' 
