# testing.py
import asyncio
import time
import sys
from redis.exceptions import ConnectionError as RedisConnectionError

REDIS_URLS = [
    "redis://127.0.0.1:6379/0",   # prefer explicit IPv4
    "redis://localhost:6379/0",   # fallback
]

async def try_async_redis(url, max_attempts=3, backoff=1.0):
    import redis.asyncio as aioredis
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[async] Attempt {attempt} -> connecting to {url}")
            r = aioredis.from_url(url, socket_connect_timeout=3, decode_responses=True)
            await r.ping()
            print(f"[async] Connected OK to {url}")
            await r.close()
            return True
        except Exception as e:
            print(f"[async] connection failed to {url}: {type(e).__name__}: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(backoff)
                backoff *= 2
    return False

def try_sync_redis(url):
    import redis
    try:
        print(f"[sync] Trying sync redis connection to {url}")
        r = redis.from_url(url, socket_connect_timeout=3, decode_responses=True)
        r.ping()
        print("[sync] Connected OK")
        return True
    except Exception as e:
        print(f"[sync] connection failed: {type(e).__name__}: {e}")
        return False

async def main():
    for url in REDIS_URLS:
        ok = await try_async_redis(url)
        if ok:
            return
    # async failed for both urls — try sync for clearer errors
    for url in REDIS_URLS:
        ok = try_sync_redis(url)
        if ok:
            return
    print("All connection attempts failed. See the error messages above.")
    print("If you are running Redis in Docker, verify port mapping and firewall settings.")
    sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
