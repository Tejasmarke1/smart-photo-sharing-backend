"""Authentication schemas."""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime
import re


class SendOTPRequest(BaseModel):
    """Request to send OTP."""
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, min_length=10, max_length=15)
    otp_type: str = Field(default="login", pattern="^(login|signup|verify_email|verify_phone)$")
    
    @validator('phone')
    def validate_phone(cls, v):
        if v and not re.match(r'^\+?[1-9]\d{9,14}$', v):
            raise ValueError('Invalid phone number format')
        return v
    
    @validator('email', 'phone', always=True)
    def check_email_or_phone(cls, v, values):
        if not values.get('email') and not v:
            raise ValueError('Either email or phone must be provided')
        return v


class SendOTPResponse(BaseModel):
    """Response after sending OTP."""
    success: bool
    message: str
    expires_in: int  # seconds
    can_resend_in: int = 60  # seconds


class VerifyOTPRequest(BaseModel):
    """Request to verify OTP."""
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    otp_code: str = Field(..., min_length=6, max_length=6, pattern="^[0-9]{6}$")


class VerifyOTPResponse(BaseModel):
    """Response after OTP verification."""
    success: bool
    message: str
    user_exists: bool
    requires_signup: bool
    temp_token: Optional[str] = None  # Temporary token for signup


class SignupRequest(BaseModel):
    """User signup request."""
    temp_token: str
    name: str = Field(..., min_length=2, max_length=255)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    profile_picture_url: Optional[str] = None
    
    # Photographer-specific fields
    business_name: Optional[str] = Field(None, max_length=255)
    studio_location: Optional[str] = Field(None, max_length=255)
    portfolio_link: Optional[str] = Field(None, max_length=512)


class SignupResponse(BaseModel):
    """Signup response with tokens."""
    success: bool
    message: str
    user: dict
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginResponse(BaseModel):
    """Login response with tokens."""
    success: bool
    message: str
    user: dict
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """Refresh token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """User profile response."""
    id: str
    name: str
    email: Optional[str]
    phone: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    profile_picture_url: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True