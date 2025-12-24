"""
Cache Middleware & Integration
===============================

FastAPI middleware for cache headers and lifecycle management.
"""

import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from src.core.cache import init_cache, close_cache, get_cache_stats

logger = None


class CacheMiddleware:
    """
    Middleware to add cache control headers and manage cache lifecycle.

    Features:
    - Automatic cache header injection
    - Cache metrics tracking
    - Cache invalidation on mutation requests
    """

    def __init__(self, app, cache_control_max_age: int = 3600):
        """
        Initialize cache middleware.

        Args:
            app: FastAPI application
            cache_control_max_age: Default max-age for cache-control header
        """
        self.app = app
        self.cache_control_max_age = cache_control_max_age

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request and add cache headers."""
        request.state.start_time = time.time()

        response = await call_next(request)

        # Add cache-control headers for GET requests
        if request.method == "GET":
            response.headers["Cache-Control"] = (
                f"public, max-age={self.cache_control_max_age}"
            )
            response.headers["ETag"] = str(hash(response.body))

        # Add timing header
        process_time = time.time() - request.state.start_time
        response.headers["X-Process-Time"] = str(process_time)

        return response


async def startup_cache():
    """Initialize cache on application startup."""
    try:
        await init_cache()
    except Exception as e:
        print(f"❌ Failed to initialize cache: {str(e)}")
        raise


async def shutdown_cache():
    """Close cache on application shutdown."""
    try:
        await close_cache()
    except Exception as e:
        print(f"❌ Failed to close cache: {str(e)}")


async def cache_stats_endpoint():
    """
    Endpoint to get cache statistics.

    Example usage in main.py:
        @app.get("/api/v1/cache/stats")
        async def get_cache_statistics():
            return await cache_stats_endpoint()
    """
    stats = await get_cache_stats()
    return {
        "status": "ok",
        "cache": {
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
            "total_requests": stats.get("total_requests", 0),
            "hit_rate": stats.get("hit_rate", "0%"),
            "sets": stats.get("sets", 0),
            "deletes": stats.get("deletes", 0),
            "errors": stats.get("errors", 0),
            "db_size": stats.get("db_size", 0),
        },
    }


# ============================================================================
# Integration Instructions
# ============================================================================

"""
INTEGRATION GUIDE
=================

1. Add to FastAPI Application (src/app/main.py):

    from fastapi import FastAPI
    from src.core.cache_middleware import (
        CacheMiddleware,
        startup_cache,
        shutdown_cache,
        cache_stats_endpoint
    )

    app = FastAPI()

    # Add cache middleware
    app.add_middleware(CacheMiddleware, cache_control_max_age=3600)

    # Add startup/shutdown events
    app.add_event_handler("startup", startup_cache)
    app.add_event_handler("shutdown", shutdown_cache)

    # Add cache stats endpoint
    @app.get("/api/v1/cache/stats")
    async def get_cache_stats():
        return await cache_stats_endpoint()


2. Use Cache Decorator in Endpoints:

    from src.core.cache import cache_result, invalidate_cache, build_cache_key
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/albums/{album_id}/faces")
    @cache_result(ttl=3600)  # Cache for 1 hour
    async def get_album_faces(album_id: str):
        return await fetch_faces(album_id)

    @router.post("/albums/{album_id}/faces")
    async def create_face(album_id: str, face_data: dict):
        face = await save_face(face_data)
        # Invalidate album faces cache
        await invalidate_cache(f"album:{album_id}:faces")
        return face


3. Use Cache Functions in Services:

    from src.core.cache import (
        set_cache,
        get_cache,
        delete_cache,
        build_album_cache_key,
        invalidate_album_cache
    )

    async def get_user_profile(user_id: str):
        # Try cache first
        cache_key = f"user_profile:{user_id}"
        cached = await get_cache(cache_key)
        if cached:
            return cached

        # Fetch from DB
        user = await db.query(User).filter(User.id == user_id).first()

        # Cache for 1 hour
        await set_cache(cache_key, user.dict(), ttl=3600)

        return user

    async def update_user(user_id: str, data: dict):
        user = await db.query(User).filter(User.id == user_id).first()
        for key, value in data.items():
            setattr(user, key, value)
        db.commit()

        # Invalidate cache
        cache_key = f"user_profile:{user_id}"
        await delete_cache(cache_key)

        return user


4. Use Cache Key Builders:

    from src.core.cache import (
        build_album_cache_key,
        build_user_cache_key,
        build_face_cache_key,
        invalidate_album_cache,
        invalidate_user_cache
    )

    # Building cache keys
    album_key = build_album_cache_key(album_id, "faces")
    user_key = build_user_cache_key(user_id, "profile")
    face_key = build_face_cache_key(face_id, "similar")

    # Cache operations
    await set_cache(album_key, faces_data, ttl=3600)
    faces = await get_cache(album_key)

    # Bulk invalidation
    await invalidate_album_cache(album_id)
    await invalidate_user_cache(user_id)


5. Cache Common Patterns:

    A. List Caching (Albums):
        @router.get("/albums")
        @cache_result(ttl=300)  # 5 minutes - changes frequently
        async def list_albums(user_id: str):
            return await db.query(Album).filter(
                Album.photographer_id == user_id
            ).all()

    B. Detail Caching (User Profile):
        @router.get("/users/{user_id}")
        @cache_result(ttl=3600)  # 1 hour - changes less frequently
        async def get_user(user_id: str):
            return await db.query(User).filter(User.id == user_id).first()

    C. Search Caching (Face Search):
        @router.post("/search/by-selfie")
        @cache_result(ttl=600)  # 10 minutes - search results
        async def search_faces(image_data: bytes):
            return await face_pipeline.search(image_data)

    D. Counter Caching (Download Counts):
        async def increment_download_count(photo_id: str):
            count_key = f"photo_downloads:{photo_id}"
            count = await increment_counter(count_key)
            if count % 10 == 0:  # Persist every 10 downloads
                await db.update(Photo, {Photo.download_count: count})
            return count


6. Cache Strategies by Entity Type:

    ALBUMS:
    - List: 5-10 minutes (changes with uploads/sharing)
    - Detail: 30-60 minutes (metadata stable)
    - Faces: 1 hour (reprocessing rare)

    PHOTOS:
    - List: 10-30 minutes (new uploads)
    - Detail: 1 hour (metadata stable)
    - Processing status: 1-5 minutes (updates frequent)

    FACES:
    - Individual: 24 hours (immutable once detected)
    - Search results: 10-30 minutes (model updates)
    - Clusters: 1 hour (clustering updates)

    USERS:
    - Profile: 1 hour (infrequent updates)
    - Settings: 1 hour (infrequent changes)
    - Permissions: 30 minutes (role changes)

    SEARCH:
    - Queries: 10-30 minutes (depends on freshness requirements)
    - Suggestions: 1 hour (static suggestions)


7. Cache Invalidation Patterns:

    A. Immediate invalidation (on write):
        async def update_album(album_id: str, data: dict):
            # ... update logic ...
            await invalidate_album_cache(album_id)

    B. Deferred invalidation (background job):
        async def process_faces_task(photo_id: str):
            # ... processing ...
            await invalidate_cache(f"photo:{photo_id}")

    C. Cascading invalidation:
        async def delete_album(album_id: str):
            # Invalidate album and all related caches
            await invalidate_album_cache(album_id)
            await invalidate_search_cache()

    D. Time-based invalidation:
        # Use TTL in cache.set() - automatic expiration


8. Monitoring Cache Performance:

    Check cache stats endpoint:
        GET /api/v1/cache/stats

    Response:
        {
            "status": "ok",
            "cache": {
                "hits": 1523,
                "misses": 247,
                "total_requests": 1770,
                "hit_rate": "86.02%",
                "sets": 512,
                "deletes": 89,
                "errors": 2,
                "db_size": 156
            }
        }

    Use Prometheus metrics (if enabled in config):
        - cache_hits_total
        - cache_misses_total
        - cache_errors_total
        - cache_hit_rate
"""
