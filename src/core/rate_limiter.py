# utils/rate_limiter.py
import os
import time
from typing import Optional, Tuple

import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
r = redis.from_url(REDIS_URL, decode_responses=True)

# Helpers / keys
def _send_key(identifier: str) -> str:
    return f"otp:send:{identifier}"


def _verify_key(identifier: str) -> str:
    return f"otp:verify:{identifier}"

def _lock_key(identifier: str) -> str:
    return f"otp:lock:{identifier}"

# Rate-limit settings (fallback to env or defaults)
OTP_SEND_LIMIT = int(os.getenv("OTP_SEND_LIMIT", "5"))
OTP_SEND_WINDOW_SECS = int(os.getenv("OTP_SEND_WINDOW_SECS", "3600"))
OTP_VERIFY_ATTEMPTS = int(os.getenv("OTP_VERIFY_ATTEMPTS", "5"))
OTP_VERIFY_LOCK_SECS = int(os.getenv("OTP_VERIFY_LOCK_SECS", "3600"))

async def can_send_otp(identifier: str) -> Tuple[bool, Optional[int]]:
    """
    Check whether we can send an OTP to `identifier` (email or phone).
    Returns (allowed, retry_after_seconds_or_None).
    """
    # If identifier is locked due to repeated verify failures, deny immediately
    lock_ttl = await r.ttl(_lock_key(identifier))
    if lock_ttl and lock_ttl > 0:
        return False, lock_ttl

    key = _send_key(identifier)
    # increment counter atomically
    current = await r.incr(key)
    if current == 1:
        # first increment, set expiry for the window
        await r.expire(key, OTP_SEND_WINDOW_SECS)

    if current > OTP_SEND_LIMIT:
        # compute TTL to include in Retry-After header
        ttl = await r.ttl(key)
        return False, ttl if ttl and ttl > 0 else OTP_SEND_WINDOW_SECS
    return True, None

async def record_failed_verify(identifier: str) -> Tuple[bool, Optional[int]]:
    """
    Record a failed OTP verification attempt for identifier.
    Returns (locked_now_bool, lock_ttl_seconds_or_None)
    """
    key = _verify_key(identifier)
    current = await r.incr(key)
    if current == 1:
        # set a TTL much longer than OTP lifetime so attempts count for some time
        # e.g., keep attempts counter for the same duration as lock window
        await r.expire(key, OTP_VERIFY_LOCK_SECS)

    if current >= OTP_VERIFY_ATTEMPTS:
        # lock identifier
        lock_key = _lock_key(identifier)
        await r.set(lock_key, "1", ex=OTP_VERIFY_LOCK_SECS)
        # optionally clear verify counter
        await r.delete(key)
        return True, OTP_VERIFY_LOCK_SECS
    return False, None

async def clear_verify_attempts(identifier: str) -> None:
    """Call on successful verification to reset counters."""
    await r.delete(_verify_key(identifier))
    await r.delete(_lock_key(identifier))
