from fastapi import APIRouter
from src.api.v1.endpoints import auth, albums, photos, uploads, faces, search, payments, persons, jobs
from src.api.v1.websockets import notifications


api_router = APIRouter()

# REST API endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(albums.router, prefix="/albums", tags=["albums"])
api_router.include_router(photos.router, prefix="/photos", tags=["photos"])
api_router.include_router(uploads.router, prefix="/uploads", tags=["uploads"])
api_router.include_router(faces.router, prefix="/faces", tags=["faces"])
api_router.include_router(persons.router, prefix="/persons", tags=["persons"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(payments.router, prefix="/payments", tags=["payments"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])

# WebSocket endpoints
api_router.include_router(notifications.router, tags=["websockets"])
