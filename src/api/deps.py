"""Dependencies for API endpoints."""
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional
from functools import wraps
from typing import Callable

from src.db.base import get_db
from src.models.user import User
from src.models.refresh_token import RefreshToken
from src.core.security import decode_token

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception
    
    if payload.get("type") != "access":
        raise credentials_exception
    
    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    return user


async def get_current_active_photographer(
    current_user: User = Depends(get_current_user)
) -> User:
    """Verify user is an active photographer."""
    from src.models.enums import UserRole
    
    if current_user.role != UserRole.photographer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only photographers can access this resource"
        )
    return current_user


async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Verify user is an admin."""
    from src.models.enums import UserRole
    
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def get_client_ip(request: Request) -> str:
    """Extract client IP address."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host if request.client else "unknown"


def get_user_agent(request: Request) -> str:
    """Extract user agent."""
    return request.headers.get("User-Agent", "unknown")


def require_roles(allowed_roles: list[str]):
    def decorator(func):
        @wraps(func)
        async def wrapper(
            *args,
            current_user=Depends(get_current_user),
            **kwargs
        ):
            # Support both ORM object and dict for tests
            role = getattr(current_user, 'role', None)
            if role is None and isinstance(current_user, dict):
                role = current_user.get('role')
            if role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator
