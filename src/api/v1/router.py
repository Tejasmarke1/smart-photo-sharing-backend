from fastapi import APIRouter
from src.api.v1.endpoints import auth, albums, photos, uploads, faces, search, payments


api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(albums.router, prefix="/albums", tags=["albums"])
api_router.include_router(photos.router, prefix="/photos", tags=["photos"])
api_router.include_router(uploads.router, prefix="/uploads", tags=["uploads"])
api_router.include_router(faces.router, prefix="/faces", tags=["faces"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(payments.router, prefix="/payments", tags=["payments"])
