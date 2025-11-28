"""Authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional

from src.db.base import get_db
from src.models.user import User
from src.models.enums import UserRole
from src.models.refresh_token import RefreshToken
from src.models.login_history import LoginHistory
from src.schemas.auth import (
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
from src.services import OTPService
from src.services import NotificationService
from src.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_temp_token,
)
from src.api.deps import get_current_user, get_client_ip, get_user_agent

router = APIRouter()


@router.post("/send-otp", response_model=SendOTPResponse)
async def send_otp(
    request: SendOTPRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Send OTP to email or phone.
    
    - Checks if user exists
    - Generates and sends OTP
    - Returns expiry time
    """
    ip_address = get_client_ip(http_request)
    user_agent = get_user_agent(http_request)
    
    # Create OTP
    otp, otp_code = OTPService.create_otp(
        db=db,
        email=request.email,
        phone=request.phone,
        otp_type=request.otp_type,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    # Send OTP via appropriate channel
    notification_service = NotificationService()
    
    if request.email:
        background_tasks.add_task(
            notification_service.send_email_otp,
            request.email,
            otp_code
        )
    elif request.phone:
        background_tasks.add_task(
            notification_service.send_sms_otp,
            request.phone,
            otp_code
        )
    
    return SendOTPResponse(
        success=True,
        message=f"OTP sent to {request.email or request.phone}",
        expires_in=300,  # 5 minutes
        can_resend_in=60
    )


@router.post("/verify-otp", response_model=VerifyOTPResponse)
async def verify_otp(
    request: VerifyOTPRequest,
    db: Session = Depends(get_db)
):
    """
    Verify OTP and check if user exists.
    
    - If user exists: Return success, user can login
    - If user doesn't exist: Return temp token for signup
    """
    # Verify OTP
    success, message = OTPService.verify_otp(
        db=db,
        email=request.email,
        phone=request.phone,
        otp_code=request.otp_code
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    # Check if user exists
    query = db.query(User)
    if request.email:
        user = query.filter(User.email == request.email).first()
    else:
        user = query.filter(User.phone == request.phone).first()
    
    if user:
        # User exists - can proceed to login
        # Generate tokens immediately
        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token_str = create_refresh_token(data={"sub": str(user.id)})
        
        # Store refresh token
        refresh_token = RefreshToken(
            user_id=user.id,
            token=refresh_token_str,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
        db.add(refresh_token)
        db.commit()
        
        # Log successful login
        login_history = LoginHistory(
            user_id=user.id,
            email=request.email,
            phone=request.phone,
            login_method="otp_email" if request.email else "otp_phone",
            is_successful=True
        )
        db.add(login_history)
        db.commit()
        
        return VerifyOTPResponse(
            success=True,
            message="Login successful",
            user_exists=True,
            requires_signup=False,
            temp_token=None
        )
    else:
        # New user - needs to signup
        temp_token = generate_temp_token()
        
        return VerifyOTPResponse(
            success=True,
            message="Please complete signup",
            user_exists=False,
            requires_signup=True,
            temp_token=temp_token
        )


@router.post("/signup", response_model=SignupResponse)
async def signup(
    request: SignupRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """
    Complete user signup.
    
    - Validates temp token
    - Creates user account
    - Returns JWT tokens
    """
    # In production, validate temp_token from Redis/cache
    # For now, we'll skip validation
    
    # Determine role based on app type (from header or request)
    app_type = http_request.headers.get("X-App-Type", "user")
    role = UserRole.photographer if app_type == "business" else UserRole.guest
    
    # Check if user already exists
    existing = db.query(User).filter(
        (User.email == request.email) | (User.phone == request.phone)
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )
    
    # Create user
    user = User(
        name=request.name,
        email=request.email,
        phone=request.phone,
        hashed_password="",  # Passwordless auth
        role=role,
        is_active=True,
        is_verified=True,  # Already verified via OTP
        profile_picture_url=request.profile_picture_url
    )
    
    # Add photographer-specific fields to metadata
    if role == UserRole.photographer and request.business_name:
        import json
        user.extra_data = json.dumps({
            "business_name": request.business_name,
            "studio_location": request.studio_location,
            "portfolio_link": request.portfolio_link
        })
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Generate tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token_str = create_refresh_token(data={"sub": str(user.id)})
    
    # Store refresh token
    refresh_token = RefreshToken(
        user_id=user.id,
        token=refresh_token_str,
        expires_at=datetime.utcnow() + timedelta(days=7),
        ip_address=get_client_ip(http_request),
        user_agent=get_user_agent(http_request)
    )
    db.add(refresh_token)
    db.commit()
    
    return SignupResponse(
        success=True,
        message="Signup successful",
        user={
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
            "phone": user.phone,
            "role": user.role.value
        },
        access_token=access_token,
        refresh_token=refresh_token_str,
        token_type="bearer"
    )


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token."""
    payload = decode_token(request.refresh_token)
    
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Verify refresh token in database
    refresh_token = db.query(RefreshToken).filter(
        RefreshToken.token == request.refresh_token
    ).first()
    
    if not refresh_token or not refresh_token.is_valid():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Update last used
    refresh_token.last_used_at = datetime.utcnow()
    
    # Generate new tokens
    user_id = str(refresh_token.user_id)
    new_access_token = create_access_token(data={"sub": user_id})
    new_refresh_token = create_refresh_token(data={"sub": user_id})
    
    # Revoke old refresh token
    
    # Create new refresh token
    new_rt = RefreshToken(
        user_id=refresh_token.user_id,
        token=new_refresh_token,
        expires_at=datetime.utcnow() + timedelta(days=7),
        ip_address=get_client_ip(http_request),
        user_agent=get_user_agent(http_request)
    )
    db.add(new_rt)
    db.commit()
    
    return RefreshTokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer"
    )


@router.post("/logout")
async def logout(
    refresh_token: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout and revoke refresh token."""
    rt = db.query(RefreshToken).filter(
        RefreshToken.token == refresh_token,
        RefreshToken.user_id == current_user.id
    ).first()
    
    if rt:
        rt.revoke()
        db.commit()
    
    return {"success": True, "message": "Logged out successfully"}


@router.post("/logout-all")
async def logout_all(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout from all devices."""
    db.query(RefreshToken).filter(
        RefreshToken.user_id == current_user.id,
        RefreshToken.is_revoked == False
    ).update({"is_revoked": True, "revoked_at": datetime.utcnow()})
    
    db.commit()
    
    return {"success": True, "message": "Logged out from all devices"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information."""
    return UserResponse(
        id=str(current_user.id),
        name=current_user.name,
        email=current_user.email,
        phone=current_user.phone,
        role=current_user.role.value,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        profile_picture_url=current_user.profile_picture_url,
        created_at=current_user.created_at
    )


@router.get("/sessions")
async def get_active_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all active sessions."""
    sessions = db.query(RefreshToken).filter(
        RefreshToken.user_id == current_user.id,
        RefreshToken.is_revoked == False
    ).all()
    
    return {
        "sessions": [
            {
                "id": str(s.id),
                "device_name": s.device_name,
                "ip_address": s.ip_address,
                "last_used": s.last_used_at,
                "created_at": s.created_at
            }
            for s in sessions
        ]
    }