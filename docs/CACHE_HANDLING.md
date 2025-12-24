# Cache Handling System Documentation

## Overview

A comprehensive, production-ready caching system for the Smart Photo Sharing backend with:

- **Redis-backed distributed caching** for multi-instance deployments
- **In-memory caching** for development
- **Async/await support** with FastAPI integration
- **TTL-based expiration** and automatic cleanup
- **Cache decorators** for easy function result caching
- **Cache invalidation strategies** for data consistency
- **Statistics and monitoring** with hit/miss ratios
- **Entity-specific cache helpers** for common operations

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Endpoints                    │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│            Cache Middleware & Decorators                │
│  (cache_result, cache invalidation, statistics)         │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│            Cache Utilities Layer                        │
│  (album, photo, face, user, search helpers)             │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│            Cache Backend (Abstract)                     │
├──────────────────────────────────────────────────────────┤
│  ┌────────────────┐      ┌────────────────────┐         │
│  │  RedisCache    │      │  InMemoryCache     │         │
│  │  (Production)  │      │  (Development)     │         │
│  └────────────────┘      └────────────────────┘         │
└─────────────────┬───────────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │  Redis Server   │
         │  (if enabled)   │
         └─────────────────┘
```

## Quick Start

### 1. Installation

Cache dependencies are already in `pyproject.toml`:

```bash
poetry install
```

### 2. Configuration

Set Redis environment variables in `.env`:

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379
CELERY_BROKER_DB=0
CELERY_RESULT_DB=1
```

For development, set `ENVIRONMENT=development` to use in-memory cache.

### 3. Application Initialization

Add to `src/app/main.py`:

```python
from fastapi import FastAPI
from src.core.cache_middleware import (
    CacheMiddleware,
    startup_cache,
    shutdown_cache,
)

app = FastAPI()

# Add cache middleware
app.add_middleware(CacheMiddleware, cache_control_max_age=3600)

# Initialize cache on startup
app.add_event_handler("startup", startup_cache)
app.add_event_handler("shutdown", shutdown_cache)

# Cache stats endpoint
@app.get("/api/v1/cache/stats")
async def get_cache_stats():
    from src.core.cache import get_cache_stats
    return {
        "status": "ok",
        "cache": await get_cache_stats()
    }
```

## Usage Patterns

### Pattern 1: Decorator-Based Caching

Cache function results automatically:

```python
from src.core.cache import cache_result

@app.get("/albums/{album_id}")
@cache_result(ttl=3600)  # Cache for 1 hour
async def get_album(album_id: str, db: Session = Depends(get_db)):
    return db.query(Album).filter(Album.id == album_id).first()
```

### Pattern 2: Manual Cache Management

Explicit cache get/set operations:

```python
from src.core.cache import get_cache, set_cache, delete_cache, build_user_cache_key

async def get_user_profile(user_id: str):
    cache_key = build_user_cache_key(user_id, "profile")
    
    # Try cache first
    cached = await get_cache(cache_key)
    if cached:
        return cached
    
    # Fetch from database
    user = await db.query(User).filter(User.id == user_id).first()
    
    # Cache for 1 hour
    await set_cache(cache_key, user.dict(), ttl=3600)
    
    return user

async def update_user_profile(user_id: str, data: dict):
    # ... update logic ...
    
    # Invalidate cache
    cache_key = build_user_cache_key(user_id, "profile")
    await delete_cache(cache_key)
```

### Pattern 3: Cache Helpers for Entities

Use pre-built helpers for common entities:

```python
from src.core.cache_utils import (
    cache_album_details,
    get_cached_album_details,
    invalidate_album_cache,
    cache_face_list,
    get_cached_face_list,
)

async def get_album_with_cache(album_id: str):
    # Try cache first
    album = await get_cached_album_details(album_id)
    if album:
        return album
    
    # Fetch from DB
    album = await db.query(Album).filter(Album.id == album_id).first()
    
    # Cache result
    await cache_album_details(album_id, album.dict())
    
    return album

async def update_album(album_id: str, data: dict):
    # ... update logic ...
    
    # Invalidate all album-related caches
    await invalidate_album_cache(album_id)
```

### Pattern 4: Search Result Caching

Cache search results with short TTL:

```python
from src.core.cache_utils import (
    cache_search_results,
    get_cached_search_results,
)

@app.post("/search/by-selfie")
async def search_by_selfie(file: UploadFile):
    query = f"selfie:{file.filename}"
    
    # Check cache
    cached_results = await get_cached_search_results(query)
    if cached_results:
        return cached_results
    
    # Perform search
    results = await face_pipeline.search(file)
    
    # Cache with 10-minute TTL
    await cache_search_results(query, results)
    
    return results
```

### Pattern 5: Counter/Statistics Caching

Track counts in cache, periodically persist to DB:

```python
from src.core.cache_utils import (
    increment_download_count,
    get_download_count,
)

async def download_photo(photo_id: str):
    # ... download logic ...
    
    # Increment in-memory counter
    count = await increment_download_count(photo_id)
    
    # Persist to DB every 10 downloads
    if count % 10 == 0:
        await db.execute(
            "UPDATE photos SET download_count = :count WHERE id = :id",
            {"count": count, "id": photo_id}
        )
```

### Pattern 6: Cache Warming

Preload cache on login:

```python
from src.core.cache_utils import warm_user_cache

async def login(email: str, password: str):
    user = await authenticate_user(email, password)
    
    # Warm cache with user data
    await warm_user_cache(
        user.id,
        {
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
            "role": user.role
        }
    )
    
    return create_tokens(user)
```

### Pattern 7: Cascading Invalidation

Invalidate related caches when data changes:

```python
async def delete_album(album_id: str):
    # Delete photos
    photos = db.query(Photo).filter(Photo.album_id == album_id).all()
    for photo in photos:
        await invalidate_photo_cache(photo.id)
    
    # Delete faces
    faces = db.query(Face).join(Photo).filter(
        Photo.album_id == album_id
    ).all()
    for face in faces:
        await invalidate_face_cache(face.id)
    
    # Delete album
    await invalidate_album_cache(album_id)
    db.query(Album).filter(Album.id == album_id).delete()
    db.commit()
```

### Pattern 8: Conditional Cache

Only set cache if conditions are met:

```python
from src.core.cache_utils import cache_if_not_exists

async def get_or_create_person(album_id: str, name: str):
    cache_key = f"person_id:{album_id}:{name}"
    
    # Try to get cached person ID
    cached_id = await get_cache(cache_key)
    if cached_id:
        return {"id": cached_id}
    
    # Check if exists
    person = db.query(Person).filter(
        Person.album_id == album_id,
        Person.name == name
    ).first()
    
    if person:
        # Cache only if we found existing person
        await cache_if_not_exists(cache_key, str(person.id), ttl=3600)
        return person
    
    # Create new
    person = Person(album_id=album_id, name=name)
    db.add(person)
    db.commit()
    
    # Cache new person
    await cache_if_not_exists(cache_key, str(person.id), ttl=3600)
    
    return person
```

## Cache Configuration by Entity Type

### Albums

```python
# List of albums (changes frequently with uploads)
TTL: 5-10 minutes
Key: user:{user_id}:albums
Use case: User's album list

# Album details (metadata stable)
TTL: 30-60 minutes
Key: album:{album_id}:details
Use case: Album info page

# Album faces (changes with processing)
TTL: 1 hour
Key: album:{album_id}:faces
Use case: Face list with filtering

# Album summary (statistics)
TTL: 1 hour
Key: album:{album_id}:summary
Use case: Dashboard stats

# Album clusters (changes with re-clustering)
TTL: 1 hour
Key: album:{album_id}:clusters
Use case: Face clustering results
```

### Photos

```python
# Photo list (new uploads frequent)
TTL: 10-30 minutes
Key: album:{album_id}:photos
Use case: Album photos grid

# Photo details (immutable metadata)
TTL: 1 hour
Key: photo:{photo_id}:details
Use case: Photo info page

# Processing status (updates frequent)
TTL: 1-5 minutes
Key: photo:{photo_id}:status
Use case: Processing progress

# Download count (statistics)
TTL: Until persistence
Key: photo:{photo_id}:downloads
Use case: Track downloads in memory
```

### Faces

```python
# Face list (after detection, stable)
TTL: 1 hour
Key: album:{album_id}:faces
Use case: Detected faces list

# Face details (immutable after detection)
TTL: 24 hours
Key: face:{face_id}:details
Use case: Face info

# Face embedding (for vector search)
TTL: 24 hours
Key: face:{face_id}:embedding
Use case: Vector similarity search

# Similar faces (search results)
TTL: 10-30 minutes
Key: face:{face_id}:similar
Use case: Find similar faces
```

### Users

```python
# User profile (infrequent updates)
TTL: 1 hour
Key: user:{user_id}:profile
Use case: User info page

# User settings (infrequent updates)
TTL: 1 hour
Key: user:{user_id}:settings
Use case: Settings page

# User permissions (role changes)
TTL: 30 minutes
Key: user:{user_id}:permissions
Use case: Access control checks
```

### Search

```python
# Search results (time-sensitive)
TTL: 10-30 minutes
Key: search:{query_hash}:results
Use case: Search result caching

# Search suggestions (static)
TTL: 1 hour
Key: search:{query_hash}:suggestions
Use case: Autocomplete suggestions
```

## Cache Invalidation Strategies

### 1. Immediate Invalidation (On Write)

```python
@router.put("/albums/{album_id}")
async def update_album(album_id: str, data: AlbumUpdate):
    # Update database
    album = await update_in_db(album_id, data)
    
    # Invalidate cache immediately
    await invalidate_album_cache(album_id)
    
    return album
```

### 2. Deferred Invalidation (Background Job)

```python
@celery_app.task
def process_faces_task(photo_id: str):
    # Process faces...
    
    # Invalidate cache after processing
    await invalidate_photo_cache(photo_id)
```

### 3. Cascading Invalidation

```python
async def update_person(person_id: str, data: dict):
    # Update person
    person = await update_in_db(person_id, data)
    
    # Invalidate cascading caches
    await invalidate_person_cache(person_id)      # Person details
    await invalidate_search_cache()                 # Search results
    # Could also invalidate album caches if person spans albums
    
    return person
```

### 4. TTL-Based Invalidation (Automatic)

Cache automatically expires based on TTL:

```python
# This will automatically expire after 1 hour
await cache_album_details(album_id, data, ttl=3600)

# TTL expires and cache miss occurs
# Next request fetches fresh data from database
```

## Monitoring & Statistics

### Get Cache Statistics

```python
@app.get("/api/v1/cache/stats")
async def get_cache_stats():
    from src.core.cache import get_cache_stats
    
    stats = await get_cache_stats()
    return {
        "hits": stats["hits"],
        "misses": stats["misses"],
        "total_requests": stats["total_requests"],
        "hit_rate": stats["hit_rate"],
        "sets": stats["sets"],
        "deletes": stats["deletes"],
        "errors": stats["errors"],
        "db_size": stats["db_size"]
    }
```

Example response:

```json
{
  "hits": 1523,
  "misses": 247,
  "total_requests": 1770,
  "hit_rate": "86.02%",
  "sets": 512,
  "deletes": 89,
  "errors": 2,
  "db_size": 156
}
```

### Performance Optimization Based on Stats

- **Low hit rate (< 70%)**: Consider increasing TTL or caching more aggressively
- **High error rate (> 0)**: Check Redis connection and error logs
- **High db_size**: May indicate memory pressure; consider reducing TTL or implementing eviction policies

## Advanced Features

### Cache Locking (Thundering Herd Prevention)

```python
from src.core.cache_utils import cache_with_lock

async def expensive_computation():
    result = await cache_with_lock(
        key="expensive_result",
        compute_func=lambda: expensive_operation(),
        ttl=3600
    )
    return result
```

### Batch Cache Operations

```python
from src.core.cache import get_many_cache, set_many_cache

# Get multiple values
values = await get_many_cache([
    "key1", "key2", "key3"
])

# Set multiple values
count = await set_many_cache({
    "key1": value1,
    "key2": value2,
    "key3": value3
}, ttl=3600)
```

### Custom Cache Key Builders

```python
def custom_key_builder(args, kwargs):
    # Extract relevant parts
    user_id = args[0]
    filters = kwargs.get("filters", {})
    
    # Build key
    filter_str = "|".join(f"{k}={v}" for k, v in filters.items())
    return f"search:{user_id}:{filter_str}"

@cache_result(ttl=600, key_builder=custom_key_builder)
async def search_albums(user_id: str, filters: dict = None):
    # Search logic
    pass
```

## Best Practices

1. **Use TTL appropriately**
   - Short TTL (1-5 min): Frequently changing data
   - Medium TTL (30-60 min): Semi-stable data
   - Long TTL (24 hours): Immutable data

2. **Invalidate strategically**
   - Immediate: Critical data consistency
   - Deferred: Background processing
   - TTL: For safety and memory management

3. **Monitor cache health**
   - Track hit rates
   - Monitor error rates
   - Check Redis memory usage

4. **Handle cache misses gracefully**
   - Always have database fallback
   - Never depend entirely on cache
   - Log cache-miss scenarios for analysis

5. **Use cache warming for critical data**
   - Warm cache on user login
   - Preload frequently accessed data
   - Reduces initial latency

6. **Batch operations when possible**
   - Use `get_many_cache` / `set_many_cache`
   - Reduces network round trips
   - Better performance

## Troubleshooting

### Redis Connection Issues

```python
# Check connection
from src.core.cache import cache
if isinstance(cache, RedisCache):
    await cache.connect()  # Will raise if connection fails
```

### Cache Not Working

```python
# Enable debug logging
import logging
logging.getLogger("src.core.cache").setLevel(logging.DEBUG)

# Check stats
stats = await get_cache_stats()
print(f"Hit rate: {stats['hit_rate']}")
```

### Memory Issues

```python
# Monitor cache size
stats = await get_cache_stats()
print(f"Cache size: {stats['db_size']} keys")

# Clear cache if needed
await clear_cache()

# Or clear by pattern
await clear_cache(pattern="search:*")
```

## Testing

```python
import pytest
from src.core.cache import InMemoryCache, set_cache, get_cache

@pytest.fixture
async def cache_fixture():
    cache = InMemoryCache()
    return cache

@pytest.mark.asyncio
async def test_cache_operations(cache_fixture):
    # Test set
    await cache_fixture.set("key", "value", ttl=3600)
    
    # Test get
    value = await cache_fixture.get("key")
    assert value == "value"
    
    # Test delete
    await cache_fixture.delete("key")
    value = await cache_fixture.get("key")
    assert value is None
```

## References

- Redis: https://redis.io/
- AsyncIO: https://docs.python.org/3/library/asyncio.html
- FastAPI: https://fastapi.tiangolo.com/
