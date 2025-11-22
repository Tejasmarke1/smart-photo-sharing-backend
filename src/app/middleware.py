# src/app/middleware.py
from fastapi import FastAPI, Request
import time
import logging

logger = logging.getLogger(__name__)

def register_middleware(app: FastAPI):

    @app.middleware("http")
    async def log_request_time(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info(f"{request.method} {request.url.path} | {duration:.2f} ms")
        return response

    return app
