"""
Cache Handling System
=====================

Comprehensive caching system with support for:
- Redis-backed distributed caching
- Async/await support
- TTL-based expiration
- Cache invalidation strategies
- Decorator-based caching for functions
- Cache statistics and monitoring
- Multi-tier caching (in-memory + Redis)
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set, Union
from uuid import UUID

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from src.app.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# Cache Configuration
# ============================================================================

CACHE_DEFAULT_TTL = 3600  # 1 hour
CACHE_LONG_TTL = 86400  # 24 hours
CACHE_SHORT_TTL = 300  # 5 minutes
CACHE_VERY_SHORT_TTL = 60  # 1 minute

# Cache key prefixes for organization
CACHE_PREFIX_USER = "user"
CACHE_PREFIX_ALBUM = "album"
CACHE_PREFIX_PHOTO = "photo"
CACHE_PREFIX_FACE = "face"
CACHE_PREFIX_PERSON = "person"
CACHE_PREFIX_SEARCH = "search"
CACHE_PREFIX_STATS = "stats"


# ============================================================================
# Cache Backend Abstract Base
# ============================================================================

class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache (optionally by pattern)."""
        pass

    @abstractmethod
    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for key (-1 if no expiry, -2 if not exists)."""
        pass

    @abstractmethod
    async def set_ttl(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        pass


# ============================================================================
# Redis Cache Backend
# ============================================================================

class RedisCache(CacheBackend):
    """Redis-backed cache backend with async support."""

    def __init__(self, redis_url: str, decode_responses: bool = True):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            decode_responses: If True, decode responses to strings
        """
        self.redis_url = redis_url
        self.decode_responses = decode_responses
        self.redis: Optional[aioredis.Redis] = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

    async def connect(self):
        """Establish Redis connection."""
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf8",
                decode_responses=self.decode_responses,
            )
            # Test connection
            await self.redis.ping()
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {str(e)}")
            self.redis = None
            raise

    async def disconnect(self):
        """Close Redis connection."""
        if self.redis:
            try:
                await self.redis.close()
                logger.info("✅ Redis cache disconnected")
            except Exception as e:
                logger.error(f"❌ Failed to disconnect Redis: {str(e)}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.redis:
            return None

        try:
            value = await self.redis.get(key)
            if value:
                self._stats["hits"] += 1
                # Try to deserialize JSON
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            else:
                self._stats["misses"] += 1
                return None
        except RedisError as e:
            logger.error(f"❌ Redis GET error for key {key}: {str(e)}")
            self._stats["errors"] += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache with optional TTL."""
        if not self.redis:
            return False

        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value, default=str)
            elif isinstance(value, (str, int, float, bool)):
                serialized = value
            else:
                serialized = json.dumps(value, default=str)

            # Set with TTL
            if ttl:
                result = await self.redis.setex(key, ttl, serialized)
            else:
                result = await self.redis.set(key, serialized)

            if result:
                self._stats["sets"] += 1
                logger.debug(f"✅ Cached {key} (ttl: {ttl}s)")
            return bool(result)

        except RedisError as e:
            logger.error(f"❌ Redis SET error for key {key}: {str(e)}")
            self._stats["errors"] += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.redis:
            return False

        try:
            result = await self.redis.delete(key)
            if result:
                self._stats["deletes"] += 1
                logger.debug(f"✅ Deleted cache key {key}")
            return bool(result)
        except RedisError as e:
            logger.error(f"❌ Redis DELETE error for key {key}: {str(e)}")
            self._stats["errors"] += 1
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis:
            return False

        try:
            return bool(await self.redis.exists(key))
        except RedisError as e:
            logger.error(f"❌ Redis EXISTS error for key {key}: {str(e)}")
            self._stats["errors"] += 1
            return False

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache (optionally by pattern)."""
        if not self.redis:
            return 0

        try:
            if pattern:
                # Clear keys matching pattern
                cursor = 0
                deleted = 0
                while True:
                    cursor, keys = await self.redis.scan(
                        cursor,
                        match=pattern,
                        count=100,
                    )
                    if keys:
                        deleted += await self.redis.delete(*keys)
                    if cursor == 0:
                        break
                logger.info(f"✅ Cleared {deleted} cache keys matching {pattern}")
                return deleted
            else:
                # Clear all
                count = await self.redis.dbsize()
                await self.redis.flushdb()
                logger.info(f"✅ Cleared entire cache ({count} keys)")
                return count

        except RedisError as e:
            logger.error(f"❌ Redis CLEAR error: {str(e)}")
            self._stats["errors"] += 1
            return 0

    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for key."""
        if not self.redis:
            return -2

        try:
            return await self.redis.ttl(key)
        except RedisError as e:
            logger.error(f"❌ Redis TTL error for key {key}: {str(e)}")
            return -2

    async def set_ttl(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        if not self.redis:
            return False

        try:
            return bool(await self.redis.expire(key, ttl))
        except RedisError as e:
            logger.error(f"❌ Redis EXPIRE error for key {key}: {str(e)}")
            self._stats["errors"] += 1
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            (self._stats["hits"] / total_requests * 100)
            if total_requests > 0
            else 0
        )

        db_size = 0
        if self.redis:
            try:
                db_size = await self.redis.dbsize()
            except RedisError:
                pass

        return {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate": f"{hit_rate:.2f}%",
            "db_size": db_size,
        }

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        if not self.redis:
            return 0

        try:
            return await self.redis.incrby(key, amount)
        except RedisError as e:
            logger.error(f"❌ Redis INCRBY error for key {key}: {str(e)}")
            self._stats["errors"] += 1
            return 0

    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement a counter."""
        if not self.redis:
            return 0

        try:
            return await self.redis.decrby(key, amount)
        except RedisError as e:
            logger.error(f"❌ Redis DECRBY error for key {key}: {str(e)}")
            self._stats["errors"] += 1
            return 0

    async def get_many(self, keys: list) -> Dict[str, Any]:
        """Get multiple keys at once."""
        if not self.redis or not keys:
            return {}

        try:
            values = await self.redis.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        result[key] = value
                self._stats["hits" if value else "misses"] += 1
            return result
        except RedisError as e:
            logger.error(f"❌ Redis MGET error: {str(e)}")
            self._stats["errors"] += 1
            return {}

    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> int:
        """Set multiple key-value pairs."""
        if not self.redis or not mapping:
            return 0

        try:
            count = 0
            for key, value in mapping.items():
                if await self.set(key, value, ttl):
                    count += 1
            return count
        except Exception as e:
            logger.error(f"❌ Error in set_many: {str(e)}")
            self._stats["errors"] += 1
            return 0


# ============================================================================
# In-Memory Cache Backend (for development/testing)
# ============================================================================

class InMemoryCache(CacheBackend):
    """Simple in-memory cache backend."""

    def __init__(self):
        """Initialize in-memory cache."""
        self.cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if expiry is None or time.time() < expiry:
                self._stats["hits"] += 1
                return value
            else:
                # Expired
                del self.cache[key]
                self._stats["misses"] += 1
                return None
        self._stats["misses"] += 1
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache."""
        expiry = (time.time() + ttl) if ttl else None
        self.cache[key] = (value, expiry)
        self._stats["sets"] += 1
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            self._stats["deletes"] += 1
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.get(key) is not None

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache."""
        if pattern:
            import fnmatch

            keys_to_delete = [k for k in self.cache if fnmatch.fnmatch(k, pattern)]
            count = len(keys_to_delete)
            for key in keys_to_delete:
                del self.cache[key]
            return count
        else:
            count = len(self.cache)
            self.cache.clear()
            return count

    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL."""
        if key not in self.cache:
            return -2
        value, expiry = self.cache[key]
        if expiry is None:
            return -1
        remaining = int(expiry - time.time())
        return max(remaining, -2)

    async def set_ttl(self, key: str, ttl: int) -> bool:
        """Set TTL for key."""
        if key not in self.cache:
            return False
        value, _ = self.cache[key]
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry)
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            (self._stats["hits"] / total_requests * 100)
            if total_requests > 0
            else 0
        )
        return {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate": f"{hit_rate:.2f}%",
            "db_size": len(self.cache),
        }


# ============================================================================
# Global Cache Instance
# ============================================================================

# Initialize cache backend based on environment
if settings.ENVIRONMENT == "development":
    cache: CacheBackend = InMemoryCache()
else:
    cache: CacheBackend = RedisCache(settings.REDIS_URL)


async def init_cache():
    """Initialize cache backend."""
    if isinstance(cache, RedisCache):
        await cache.connect()


async def close_cache():
    """Close cache backend."""
    if isinstance(cache, RedisCache):
        await cache.disconnect()


# ============================================================================
# Cache Key Builders
# ============================================================================


def build_cache_key(*parts: Union[str, int, UUID]) -> str:
    """Build a cache key from parts."""
    str_parts = [str(p) for p in parts]
    return ":".join(str_parts)


def build_user_cache_key(user_id: Union[str, UUID], suffix: str = "") -> str:
    """Build cache key for user."""
    return build_cache_key(CACHE_PREFIX_USER, str(user_id), suffix)


def build_album_cache_key(album_id: Union[str, UUID], suffix: str = "") -> str:
    """Build cache key for album."""
    return build_cache_key(CACHE_PREFIX_ALBUM, str(album_id), suffix)


def build_photo_cache_key(photo_id: Union[str, UUID], suffix: str = "") -> str:
    """Build cache key for photo."""
    return build_cache_key(CACHE_PREFIX_PHOTO, str(photo_id), suffix)


def build_face_cache_key(face_id: Union[str, UUID], suffix: str = "") -> str:
    """Build cache key for face."""
    return build_cache_key(CACHE_PREFIX_FACE, str(face_id), suffix)


def build_person_cache_key(person_id: Union[str, UUID], suffix: str = "") -> str:
    """Build cache key for person."""
    return build_cache_key(CACHE_PREFIX_PERSON, str(person_id), suffix)


def build_search_cache_key(query: str, suffix: str = "") -> str:
    """Build cache key for search results."""
    import hashlib

    query_hash = hashlib.md5(query.encode()).hexdigest()
    return build_cache_key(CACHE_PREFIX_SEARCH, query_hash, suffix)


def build_stats_cache_key(stat_type: str, suffix: str = "") -> str:
    """Build cache key for statistics."""
    return build_cache_key(CACHE_PREFIX_STATS, stat_type, suffix)


# ============================================================================
# Decorator-Based Caching
# ============================================================================


def cache_result(
    ttl: int = CACHE_DEFAULT_TTL,
    key_builder: Optional[Callable] = None,
):
    """
    Decorator to cache async function results.

    Args:
        ttl: Time-to-live in seconds
        key_builder: Optional custom key builder function

    Example:
        @cache_result(ttl=3600)
        async def get_user(user_id: str):
            return await db.query(User).filter(User.id == user_id).first()

        @cache_result(ttl=300, key_builder=lambda args: f"album:{args[0]}")
        async def get_album_faces(album_id: str):
            return await db.query(Face).filter(Face.album_id == album_id).all()
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(args, kwargs)
            else:
                # Default: use function name and first arg
                if args:
                    cache_key = build_cache_key(func.__name__, str(args[0]))
                else:
                    cache_key = build_cache_key(func.__name__, str(kwargs))

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"🎯 Cache hit: {cache_key}")
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Cache result
            if result is not None:
                await cache.set(cache_key, result, ttl)
                logger.debug(f"💾 Cached result: {cache_key} (ttl: {ttl}s)")

            return result

        return wrapper

    return decorator


# ============================================================================
# Cache Invalidation
# ============================================================================


async def invalidate_cache(*keys: str) -> int:
    """
    Invalidate cache for specific keys.

    Args:
        keys: One or more cache keys to invalidate

    Returns:
        Number of keys deleted

    Example:
        await invalidate_cache(f"album:{album_id}")
        await invalidate_cache(f"user:{user_id}", f"stats:*")
    """
    deleted = 0
    for key in keys:
        if "*" in key:
            deleted += await cache.clear(key)
        else:
            if await cache.delete(key):
                deleted += 1

    if deleted > 0:
        logger.info(f"🗑️  Invalidated {deleted} cache keys")

    return deleted


async def invalidate_album_cache(album_id: Union[str, UUID]) -> int:
    """Invalidate all cache for an album."""
    pattern = build_cache_key(CACHE_PREFIX_ALBUM, str(album_id), "*")
    return await invalidate_cache(pattern)


async def invalidate_user_cache(user_id: Union[str, UUID]) -> int:
    """Invalidate all cache for a user."""
    pattern = build_cache_key(CACHE_PREFIX_USER, str(user_id), "*")
    return await invalidate_cache(pattern)


async def invalidate_photo_cache(photo_id: Union[str, UUID]) -> int:
    """Invalidate all cache for a photo."""
    pattern = build_cache_key(CACHE_PREFIX_PHOTO, str(photo_id), "*")
    return await invalidate_cache(pattern)


async def invalidate_face_cache(face_id: Union[str, UUID]) -> int:
    """Invalidate all cache for a face."""
    pattern = build_cache_key(CACHE_PREFIX_FACE, str(face_id), "*")
    return await invalidate_cache(pattern)


async def invalidate_search_cache() -> int:
    """Invalidate all search cache."""
    pattern = build_cache_key(CACHE_PREFIX_SEARCH, "*")
    return await invalidate_cache(pattern)


# ============================================================================
# Cache Statistics & Monitoring
# ============================================================================


async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return await cache.get_stats()


async def clear_cache(pattern: Optional[str] = None) -> int:
    """Clear entire cache or by pattern."""
    return await cache.clear(pattern)


# ============================================================================
# Cache Operations
# ============================================================================


async def set_cache(
    key: str,
    value: Any,
    ttl: int = CACHE_DEFAULT_TTL,
) -> bool:
    """Set cache value."""
    return await cache.set(key, value, ttl)


async def get_cache(key: str) -> Optional[Any]:
    """Get cache value."""
    return await cache.get(key)


async def delete_cache(key: str) -> bool:
    """Delete cache value."""
    return await cache.delete(key)


async def cache_exists(key: str) -> bool:
    """Check if cache key exists."""
    return await cache.exists(key)


async def get_cache_ttl(key: str) -> int:
    """Get remaining TTL for cache key."""
    return await cache.get_ttl(key)


async def set_cache_ttl(key: str, ttl: int) -> bool:
    """Set TTL for cache key."""
    return await cache.set_ttl(key, ttl)


async def increment_counter(key: str, amount: int = 1) -> int:
    """Increment a counter in cache."""
    if isinstance(cache, RedisCache):
        return await cache.increment(key, amount)
    else:
        current = await cache.get(key) or 0
        new_value = current + amount
        await cache.set(key, new_value)
        return new_value


async def decrement_counter(key: str, amount: int = 1) -> int:
    """Decrement a counter in cache."""
    if isinstance(cache, RedisCache):
        return await cache.decrement(key, amount)
    else:
        current = await cache.get(key) or 0
        new_value = max(0, current - amount)
        await cache.set(key, new_value)
        return new_value


async def get_many_cache(keys: list) -> Dict[str, Any]:
    """Get multiple cache values."""
    if isinstance(cache, RedisCache):
        return await cache.get_many(keys)
    else:
        result = {}
        for key in keys:
            value = await cache.get(key)
            if value is not None:
                result[key] = value
        return result


async def set_many_cache(
    mapping: Dict[str, Any],
    ttl: int = CACHE_DEFAULT_TTL,
) -> int:
    """Set multiple cache values."""
    if isinstance(cache, RedisCache):
        return await cache.set_many(mapping, ttl)
    else:
        count = 0
        for key, value in mapping.items():
            if await cache.set(key, value, ttl):
                count += 1
        return count
