"""
Authentication schemas package.

This package contains all request/response models used for
authentication, OTP verification, signup, login, and token refresh flows.
"""

from .auth import (
    SendOTPRequest,
    SendOTPResponse,
    VerifyOTPRequest,
    VerifyOTPResponse,
    SignupRequest,
    SignupResponse,
    LoginResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    UserResponse,
)

__all__ = [
    "SendOTPRequest",
    "SendOTPResponse",
    "VerifyOTPRequest",
    "VerifyOTPResponse",
    "SignupRequest",
    "SignupResponse",
    "LoginResponse",
    "RefreshTokenRequest",
    "RefreshTokenResponse",
    "UserResponse",
]
