"""Security utilities for authentication."""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets
import random
import hashlib
import logging
from passlib.hash import bcrypt

from src.app.config import settings

# Use bcrypt_sha256 to avoid bcrypt's 72-byte input limit.
# Keep "bcrypt" in the list so older bcrypt-only hashes still verify.
pwd_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")


def generate_otp(length: int = 6) -> str:
    """Generate random OTP."""
    return ''.join([str(random.randint(0, 9)) for _ in range(length)])


def generate_temp_token() -> str:
    """Generate temporary token for signup flow."""
    return secrets.token_urlsafe(32)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


# at top of file under imports


logger = logging.getLogger("uvicorn.error")
print(">>> SECURITY MODULE LOADED (src/core/security.py) <<<")  # confirm file loaded



def hash_password(password: str) -> str:
    """
    Safe hash: try CryptContext (bcrypt_sha256 preferred), fallback to sha256+bcrypt
    to avoid bcrypt 72-byte issues.
    """
    logger.info("hash_password called; type=%s", type(password))
    if not isinstance(password, str):
        logger.error("hash_password: password is not str (type=%s)", type(password))
        raise ValueError("Password must be a string")
    if password == "":
        raise ValueError("Password cannot be empty")

    pw_bytes = password.encode("utf-8")
    logger.info("hash_password: password byte-length=%d", len(pw_bytes))

    # Primary: let passlib handle it (bcrypt_sha256 -> avoids 72 byte issue)
    try:
        return pwd_context.hash(password)
    except Exception as exc:
        # Log and fall back to safe pre-hash with SHA256 + bcrypt
        logger.exception("pwd_context.hash failed; falling back to sha256+bcrypt: %s", exc)
        pre = hashlib.sha256(pw_bytes).digest()
        return bcrypt.hash(pre)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify and try fallback verification if needed.
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as exc:
        logger.warning("pwd_context.verify failed (%s); trying sha256-prehash verify", exc)
        pre = hashlib.sha256(plain_password.encode("utf-8")).digest()
        try:
            return bcrypt.verify(pre, hashed_password)
        except Exception:
            return False


def needs_rehash(hashed_password: str) -> bool:
    return pwd_context.needs_update(hashed_password)

