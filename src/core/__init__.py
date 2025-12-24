"""
Core package initializer.

This package provides core utilities such as authentication helpers,
security functions, token handling, password hashing, and caching.
"""

from .security import (
    generate_otp,
    generate_temp_token,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

from .cache import (
    init_cache,
    close_cache,
    cache_result,
    invalidate_cache,
    set_cache,
    get_cache,
    delete_cache,
    cache_exists,
    build_cache_key,
    build_album_cache_key,
    build_user_cache_key,
    build_photo_cache_key,
    build_face_cache_key,
    build_person_cache_key,
    build_search_cache_key,
)

__all__ = [
    # Security
    "generate_otp",
    "generate_temp_token",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "hash_password",
    "verify_password",
    # Cache
    "init_cache",
    "close_cache",
    "cache_result",
    "invalidate_cache",
    "set_cache",
    "get_cache",
    "delete_cache",
    "cache_exists",
    "build_cache_key",
    "build_album_cache_key",
    "build_user_cache_key",
    "build_photo_cache_key",
    "build_face_cache_key",
    "build_person_cache_key",
    "build_search_cache_key",
]
