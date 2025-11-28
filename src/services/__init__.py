"""
Services package initializer.

Re-exports important service classes/packets so callers can import from
`src.services` instead of deep module paths.
"""

# Re-export messaging subpackage services for convenience
from .messaging import NotificationService, OTPService

__all__ = [
    "NotificationService",
    "OTPService",
]
