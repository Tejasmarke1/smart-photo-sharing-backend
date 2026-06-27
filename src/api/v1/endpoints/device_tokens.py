"""Device token endpoints for push notification registration."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID

from src.db.base import get_db
from src.api.deps import get_current_user
from src.models.user import User
from src.models.device_token import DeviceToken
from src.schemas.notification import DeviceTokenRegisterRequest, DeviceTokenResponse

router = APIRouter()


@router.post("/register", response_model=DeviceTokenResponse, status_code=status.HTTP_201_CREATED)
async def register_device_token(
    payload: DeviceTokenRegisterRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Register or update a device token for push notifications.
    If the token already exists, update the user association and reactivate it.
    """
    # Check if token already exists
    existing = db.query(DeviceToken).filter(DeviceToken.token == payload.token).first()
    
    if existing:
        existing.user_id = current_user.id
        existing.device_type = payload.device_type.value
        existing.device_name = payload.device_name
        existing.is_active = True
        db.commit()
        db.refresh(existing)
        return existing
    
    device_token = DeviceToken(
        user_id=current_user.id,
        token=payload.token,
        device_type=payload.device_type.value,
        device_name=payload.device_name,
        is_active=True,
    )
    db.add(device_token)
    db.commit()
    db.refresh(device_token)
    return device_token


@router.delete("/{token}", status_code=status.HTTP_204_NO_CONTENT)
async def unregister_device_token(
    token: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Unregister a device token (e.g., on logout).
    Deactivates the token rather than deleting it.
    """
    device_token = db.query(DeviceToken).filter(
        DeviceToken.token == token,
        DeviceToken.user_id == current_user.id
    ).first()
    
    if not device_token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device token not found"
        )
    
    device_token.is_active = False
    db.commit()
    return None
