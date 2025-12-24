"""
Cache Utilities & Use-Case Specific Helpers
=============================================

Ready-to-use cache functions for specific domain entities and operations.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.core.cache import (
    cache_result,
    delete_cache,
    get_cache,
    set_cache,
    invalidate_cache,
    build_cache_key,
    build_album_cache_key,
    build_user_cache_key,
    build_photo_cache_key,
    build_face_cache_key,
    build_person_cache_key,
    build_search_cache_key,
    increment_counter,
    CACHE_DEFAULT_TTL,
    CACHE_LONG_TTL,
    CACHE_SHORT_TTL,
    CACHE_VERY_SHORT_TTL,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Album Cache Helpers
# ============================================================================


async def cache_album_list(user_id: UUID, albums: List[Dict]) -> None:
    """Cache user's album list."""
    key = build_user_cache_key(user_id, "albums")
    await set_cache(key, albums, CACHE_SHORT_TTL)
    logger.debug(f"Cached album list for user {user_id}")


async def get_cached_album_list(user_id: UUID) -> Optional[List[Dict]]:
    """Get cached album list."""
    key = build_user_cache_key(user_id, "albums")
    return await get_cache(key)


async def cache_album_details(album_id: UUID, album_data: Dict) -> None:
    """Cache album details."""
    key = build_album_cache_key(album_id, "details")
    await set_cache(key, album_data, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached album details for {album_id}")


async def get_cached_album_details(album_id: UUID) -> Optional[Dict]:
    """Get cached album details."""
    key = build_album_cache_key(album_id, "details")
    return await get_cache(key)


async def invalidate_album_details(album_id: UUID) -> int:
    """Invalidate album details cache."""
    pattern = build_album_cache_key(album_id, "*")
    return await invalidate_cache(pattern)


# ============================================================================
# Photo Cache Helpers
# ============================================================================


async def cache_photo_list(album_id: UUID, photos: List[Dict]) -> None:
    """Cache photos in album."""
    key = build_album_cache_key(album_id, "photos")
    await set_cache(key, photos, CACHE_SHORT_TTL)
    logger.debug(f"Cached {len(photos)} photos for album {album_id}")


async def get_cached_photo_list(album_id: UUID) -> Optional[List[Dict]]:
    """Get cached photos."""
    key = build_album_cache_key(album_id, "photos")
    return await get_cache(key)


async def cache_photo_details(photo_id: UUID, photo_data: Dict) -> None:
    """Cache photo details."""
    key = build_photo_cache_key(photo_id, "details")
    await set_cache(key, photo_data, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached photo details for {photo_id}")


async def get_cached_photo_details(photo_id: UUID) -> Optional[Dict]:
    """Get cached photo details."""
    key = build_photo_cache_key(photo_id, "details")
    return await get_cache(key)


async def cache_photo_processing_status(
    photo_id: UUID,
    status: Dict,
) -> None:
    """Cache photo processing status (short TTL as it changes frequently)."""
    key = build_photo_cache_key(photo_id, "status")
    await set_cache(key, status, CACHE_VERY_SHORT_TTL)


async def get_cached_photo_processing_status(photo_id: UUID) -> Optional[Dict]:
    """Get cached processing status."""
    key = build_photo_cache_key(photo_id, "status")
    return await get_cache(key)


async def invalidate_photo_cache(photo_id: UUID, album_id: Optional[UUID] = None) -> int:
    """Invalidate photo cache and related album cache."""
    deleted = 0
    # Invalidate photo
    deleted += await invalidate_cache(build_photo_cache_key(photo_id, "*"))
    # Invalidate album photo list
    if album_id:
        deleted += await invalidate_cache(build_album_cache_key(album_id, "photos"))
    return deleted


# ============================================================================
# Face Cache Helpers
# ============================================================================


async def cache_face_list(album_id: UUID, faces: List[Dict]) -> None:
    """Cache faces in album."""
    key = build_album_cache_key(album_id, "faces")
    await set_cache(key, faces, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached {len(faces)} faces for album {album_id}")


async def get_cached_face_list(album_id: UUID) -> Optional[List[Dict]]:
    """Get cached faces."""
    key = build_album_cache_key(album_id, "faces")
    return await get_cache(key)


async def cache_face_details(face_id: UUID, face_data: Dict) -> None:
    """Cache face details (immutable after detection)."""
    key = build_face_cache_key(face_id, "details")
    await set_cache(key, face_data, CACHE_LONG_TTL)
    logger.debug(f"Cached face details for {face_id}")


async def get_cached_face_details(face_id: UUID) -> Optional[Dict]:
    """Get cached face details."""
    key = build_face_cache_key(face_id, "details")
    return await get_cache(key)


async def cache_similar_faces(
    face_id: UUID,
    similar_faces: List[Dict],
) -> None:
    """Cache similar faces search results."""
    key = build_face_cache_key(face_id, "similar")
    await set_cache(key, similar_faces, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached similar faces for {face_id}")


async def get_cached_similar_faces(face_id: UUID) -> Optional[List[Dict]]:
    """Get cached similar faces."""
    key = build_face_cache_key(face_id, "similar")
    return await get_cache(key)


async def cache_face_clusters(album_id: UUID, clusters: Dict) -> None:
    """Cache face clustering results."""
    key = build_album_cache_key(album_id, "clusters")
    await set_cache(key, clusters, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached clusters for album {album_id}")


async def get_cached_face_clusters(album_id: UUID) -> Optional[Dict]:
    """Get cached clusters."""
    key = build_album_cache_key(album_id, "clusters")
    return await get_cache(key)


async def invalidate_face_cache(face_id: UUID, album_id: Optional[UUID] = None) -> int:
    """Invalidate face cache and search results."""
    deleted = 0
    # Invalidate face
    deleted += await invalidate_cache(build_face_cache_key(face_id, "*"))
    # Invalidate search results
    deleted += await invalidate_cache(build_search_cache_key("*"))
    # Invalidate album faces
    if album_id:
        deleted += await invalidate_cache(build_album_cache_key(album_id, "faces"))
        deleted += await invalidate_cache(build_album_cache_key(album_id, "clusters"))
    return deleted


# ============================================================================
# Person Cache Helpers
# ============================================================================


async def cache_person_details(person_id: UUID, person_data: Dict) -> None:
    """Cache person details."""
    key = build_person_cache_key(person_id, "details")
    await set_cache(key, person_data, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached person details for {person_id}")


async def get_cached_person_details(person_id: UUID) -> Optional[Dict]:
    """Get cached person details."""
    key = build_person_cache_key(person_id, "details")
    return await get_cache(key)


async def cache_person_faces(person_id: UUID, faces: List[Dict]) -> None:
    """Cache faces for a person."""
    key = build_person_cache_key(person_id, "faces")
    await set_cache(key, faces, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached {len(faces)} faces for person {person_id}")


async def get_cached_person_faces(person_id: UUID) -> Optional[List[Dict]]:
    """Get cached person faces."""
    key = build_person_cache_key(person_id, "faces")
    return await get_cache(key)


async def invalidate_person_cache(person_id: UUID) -> int:
    """Invalidate person cache."""
    pattern = build_person_cache_key(person_id, "*")
    return await invalidate_cache(pattern)


# ============================================================================
# User Cache Helpers
# ============================================================================


async def cache_user_profile(user_id: UUID, profile_data: Dict) -> None:
    """Cache user profile."""
    key = build_user_cache_key(user_id, "profile")
    await set_cache(key, profile_data, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached profile for user {user_id}")


async def get_cached_user_profile(user_id: UUID) -> Optional[Dict]:
    """Get cached user profile."""
    key = build_user_cache_key(user_id, "profile")
    return await get_cache(key)


async def cache_user_settings(user_id: UUID, settings_data: Dict) -> None:
    """Cache user settings."""
    key = build_user_cache_key(user_id, "settings")
    await set_cache(key, settings_data, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached settings for user {user_id}")


async def get_cached_user_settings(user_id: UUID) -> Optional[Dict]:
    """Get cached user settings."""
    key = build_user_cache_key(user_id, "settings")
    return await get_cache(key)


async def cache_user_permissions(user_id: UUID, permissions: List[str]) -> None:
    """Cache user permissions."""
    key = build_user_cache_key(user_id, "permissions")
    await set_cache(key, permissions, CACHE_SHORT_TTL)
    logger.debug(f"Cached permissions for user {user_id}")


async def get_cached_user_permissions(user_id: UUID) -> Optional[List[str]]:
    """Get cached user permissions."""
    key = build_user_cache_key(user_id, "permissions")
    return await get_cache(key)


async def invalidate_user_cache(user_id: UUID) -> int:
    """Invalidate all user cache."""
    pattern = build_user_cache_key(user_id, "*")
    return await invalidate_cache(pattern)


# ============================================================================
# Search Cache Helpers
# ============================================================================


async def cache_search_results(
    query: str,
    results: List[Dict],
) -> None:
    """Cache search results."""
    key = build_search_cache_key(query, "results")
    await set_cache(key, results, CACHE_SHORT_TTL)
    logger.debug(f"Cached search results for '{query}'")


async def get_cached_search_results(query: str) -> Optional[List[Dict]]:
    """Get cached search results."""
    key = build_search_cache_key(query, "results")
    return await get_cache(key)


async def cache_search_suggestions(
    query: str,
    suggestions: List[str],
) -> None:
    """Cache search suggestions."""
    key = build_search_cache_key(query, "suggestions")
    await set_cache(key, suggestions, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached suggestions for '{query}'")


async def get_cached_search_suggestions(query: str) -> Optional[List[str]]:
    """Get cached search suggestions."""
    key = build_search_cache_key(query, "suggestions")
    return await get_cache(key)


# ============================================================================
# Statistics & Metrics Cache
# ============================================================================


async def increment_download_count(photo_id: UUID) -> int:
    """
    Increment download count for photo.

    Downloads are tracked in cache for performance, then persisted
    periodically to database.
    """
    key = build_photo_cache_key(photo_id, "downloads")
    count = await increment_counter(key)
    logger.debug(f"Download count for photo {photo_id}: {count}")
    return count


async def increment_view_count(photo_id: UUID) -> int:
    """Increment view count for photo."""
    key = build_photo_cache_key(photo_id, "views")
    count = await increment_counter(key)
    logger.debug(f"View count for photo {photo_id}: {count}")
    return count


async def increment_share_count(album_id: UUID) -> int:
    """Increment share count for album."""
    key = build_album_cache_key(album_id, "shares")
    count = await increment_counter(key)
    logger.debug(f"Share count for album {album_id}: {count}")
    return count


async def get_download_count(photo_id: UUID) -> int:
    """Get cached download count."""
    key = build_photo_cache_key(photo_id, "downloads")
    count = await get_cache(key)
    return count or 0


async def get_view_count(photo_id: UUID) -> int:
    """Get cached view count."""
    key = build_photo_cache_key(photo_id, "views")
    count = await get_cache(key)
    return count or 0


async def get_share_count(album_id: UUID) -> int:
    """Get cached share count."""
    key = build_album_cache_key(album_id, "shares")
    count = await get_cache(key)
    return count or 0


# ============================================================================
# Batch Operations
# ============================================================================


async def cache_album_summary(album_id: UUID, summary: Dict) -> None:
    """
    Cache album summary with multiple statistics.

    Example summary:
        {
            "photo_count": 150,
            "face_count": 342,
            "person_count": 23,
            "download_count": 45,
            "share_count": 12,
            "storage_used": "2.5GB"
        }
    """
    key = build_album_cache_key(album_id, "summary")
    await set_cache(key, summary, CACHE_DEFAULT_TTL)
    logger.debug(f"Cached album summary for {album_id}")


async def get_cached_album_summary(album_id: UUID) -> Optional[Dict]:
    """Get cached album summary."""
    key = build_album_cache_key(album_id, "summary")
    return await get_cache(key)


async def cache_face_embedding(face_id: UUID, embedding: List[float]) -> None:
    """
    Cache face embedding vector.

    Note: For production, consider using a specialized vector database
    like Pinecone, Weaviate, or Qdrant instead of general-purpose cache.
    """
    key = build_face_cache_key(face_id, "embedding")
    await set_cache(key, embedding, CACHE_LONG_TTL)


async def get_cached_face_embedding(face_id: UUID) -> Optional[List[float]]:
    """Get cached face embedding."""
    key = build_face_cache_key(face_id, "embedding")
    return await get_cache(key)


# ============================================================================
# Cache Warming (Preloading)
# ============================================================================


async def warm_user_cache(user_id: UUID, profile_data: Dict) -> None:
    """
    Warm up user cache with profile data on login.

    This proactively loads data that will likely be requested,
    improving perceived performance.
    """
    logger.info(f"Warming cache for user {user_id}")

    await set_cache(
        build_user_cache_key(user_id, "profile"),
        profile_data,
        CACHE_DEFAULT_TTL,
    )
    logger.debug(f"Warmed cache for user {user_id}")


async def warm_album_cache(album_id: UUID, album_data: Dict) -> None:
    """Warm up album cache with details and metadata."""
    logger.info(f"Warming cache for album {album_id}")

    await set_cache(
        build_album_cache_key(album_id, "details"),
        album_data,
        CACHE_DEFAULT_TTL,
    )
    logger.debug(f"Warmed cache for album {album_id}")


# ============================================================================
# Conditional Cache Operations
# ============================================================================


async def cache_if_not_exists(key: str, value: Any, ttl: int = CACHE_DEFAULT_TTL) -> bool:
    """
    Set cache value only if key doesn't already exist.

    Returns True if set, False if already exists.
    """
    if await get_cache(key) is None:
        return await set_cache(key, value, ttl)
    return False


async def cache_with_lock(
    key: str,
    compute_func,
    ttl: int = CACHE_DEFAULT_TTL,
) -> Any:
    """
    Cache with locking to prevent thundering herd.

    If cache miss, acquire lock and compute value once.
    """
    # Try cache first
    cached = await get_cache(key)
    if cached is not None:
        return cached

    # Use a lock key to prevent multiple concurrent computations
    lock_key = f"{key}:lock"
    max_retries = 10
    retry_count = 0

    while retry_count < max_retries:
        # Try to acquire lock
        if await cache_if_not_exists(lock_key, "locked", ttl=5):
            try:
                # Compute value
                value = await compute_func()
                # Cache result
                await set_cache(key, value, ttl)
                return value
            finally:
                # Release lock
                await delete_cache(lock_key)

        # Wait and retry
        import asyncio

        await asyncio.sleep(0.1)
        retry_count += 1

        # Try to get computed value
        cached = await get_cache(key)
        if cached is not None:
            return cached

    # Fallback: compute without cache
    logger.warning(f"Cache lock timeout for key {key}, computing without cache")
    return await compute_func()
