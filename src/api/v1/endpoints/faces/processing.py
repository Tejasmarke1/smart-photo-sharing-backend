"""
Face Detection and Search API Endpoints
========================================

Production-grade FastAPI endpoints for face detection, labeling, and search.

Features:
- Face detection and embedding generation
- Face labeling and person management
- Selfie-based search
- Album-level clustering
- Quality assessment and filtering
- Comprehensive error handling
- Rate limiting and caching
- OpenAPI documentation
"""

from typing import List, Optional, Dict, Any
import json
from uuid import UUID
import logging
from datetime import datetime

from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    status, 
    UploadFile, 
    File,
    Query,
    Body,
    BackgroundTasks
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import numpy as np
import cv2
from pydantic import BaseModel, Field, validator

from src.db.base import get_db
from src.models.face import Face
from src.models.face_person import FacePerson
from src.models.person import Person
from src.models.photo import Photo
from src.models.album import Album
from src.models.face_cluster import FaceCluster
from src.api.deps import get_current_user, require_roles
from src.models.user import User
from src.services.face.pipeline import FacePipeline, create_pipeline
from src.services.storage.s3 import S3Service
from src.schemas.face import (
    FaceResponse,
    FaceLabelRequest,
    FaceSearchRequest,
    FaceSearchResponse,
    FaceClusterRequest,
    FaceQualityFilter,
    FaceListResult,
    JobAccepted,
    ReprocessResponse,
    BatchReprocessRequest,
    BatchDeleteRequest,
    DuplicateFace,
    OutlierFace,
    FaceQualityResponse,
    AlbumDetectRequest,
    AdvancedFilterRequest,
    QualityCheckRequest,
    AutoClusterRequest,
    ClusterStatusResponse,
    ClusterDetailResponse,
    ClusterResultsResponse,
    ClusterReviewRequest,
    ClusterAcceptRequest,
    ClusterSplitRequest,
    ClusterMergeRequest,
    AutoLabelRequest,
    LabelSuggestion,
    LabelSuggestionsResponse,
    ConfirmLabelRequest,
    AutoLabelResponse
)
from src.tasks.workers.face_processor import process_faces_task, process_album_photos_task, cluster_album_task
from src.core.cache import cache_result, invalidate_cache
from src.core.rate_limiter import rate_limit
from src.utils.validators import validate_image

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()


# Shared helpers for faces endpoints
from src.api.v1.endpoints.faces.helpers import get_pipeline, get_face_or_404, serialize_face

router = APIRouter()

@router.post(
    "/photos/{photo_id}/detect",
    response_model=JobAccepted,
    summary="Detect faces in photo",
    description="Manually trigger face detection for a photo",
    status_code=status.HTTP_202_ACCEPTED
)
@require_roles(['photographer', 'admin','guest'])
async def detect_faces_in_photo(
    photo_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Manually trigger face detection for a photo.
    
    Useful for:
    - Reprocessing failed photos
    - Re-detecting with updated models
    - Testing/debugging
    """
    # Verify photo exists and user has access
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photo {photo_id} not found"
        )
    
    album = db.query(Album).filter(Album.id == photo.album_id).first()
    # Permission check: owner or admin
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if album and album.photographer_id not in (user_id,) and role != 'admin':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    
    # Enqueue face processing task
    task_id = process_faces_task.delay(str(photo_id))
    
    logger.info(f"Enqueued face detection task {task_id} for photo {photo_id}")
    
    return {
        "message": "Face detection job started",
        "job_id": str(task_id),
        "photo_id": str(photo_id),
        "status_url": f"/api/v1/jobs/{task_id}"
    }


@router.post(
    "/albums/{album_id}/detect",
    response_model=JobAccepted,
    summary="Detect faces across an album",
    description="Enqueue face detection for all photos in an album (or a provided subset)",
    status_code=status.HTTP_202_ACCEPTED
)
@require_roles(['photographer', 'admin','guest'])
async def detect_faces_in_album(
    album_id: UUID,
    request: AlbumDetectRequest = Body(default_factory=AlbumDetectRequest),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Trigger face detection for every photo in an album (or a specified subset)."""
    album = db.query(Album).filter(Album.id == album_id).first()
    if not album:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Album {album_id} not found")

    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if album.photographer_id not in (user_id,) and role != 'admin':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    photo_ids = [str(pid) for pid in request.photo_ids] if request.photo_ids else None
    task_id = process_album_photos_task.delay(str(album_id), photo_ids)

    logger.info(f"Enqueued album face detection task {task_id} for album {album_id} (photos={len(photo_ids) if photo_ids else 'all'})")

    return {
        "message": "Album face detection job started",
        "job_id": str(task_id),
        "album_id": str(album_id),
        "status_url": f"/api/v1/jobs/{task_id}"
    }


@router.post(
    "/faces/{face_id}/reprocess",
    response_model=ReprocessResponse,
    summary="Reprocess single face",
    description="Re-run face detection and embedding for a specific face",
    status_code=status.HTTP_202_ACCEPTED
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="reprocess_face", max_calls=20, period=60)
async def reprocess_face(
    face_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Reprocess a single face detection.
    
    Use when:
    - Face quality is poor
    - Detection seems incorrect
    - Embedding is missing
    - Using updated model
    """
    face = await get_face_or_404(face_id, db, current_user)
    
    # Get photo for reprocessing
    photo = db.query(Photo).filter(Photo.id == face.photo_id).first()
    if not photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Photo not found"
        )
    
    # Enqueue reprocessing task
    task_id = process_faces_task.delay(str(photo.id))
    
    logger.info(f"Enqueued face reprocessing task {task_id} for face {face_id} (photo {photo.id})")
    
    return ReprocessResponse(
        message="Face reprocessing started",
        job_id=str(task_id),
        face_ids=[face_id],
        status_url=f"/api/v1/jobs/{task_id}"
    )




@router.post(
    "/albums/{album_id}/faces/reprocess",
    response_model=ReprocessResponse,
    summary="Reprocess album faces",
    description="Re-run face detection for all photos in an album",
    status_code=status.HTTP_202_ACCEPTED
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="reprocess_album", max_calls=10, period=60)
async def reprocess_album_faces(
    album_id: UUID,
    request: BatchReprocessRequest = Body(default_factory=BatchReprocessRequest),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Reprocess all faces in an album.
    
    Options:
    - Reprocess only low-quality faces
    - Force reprocess all faces
    
    Useful for:
    - Upgrading to new detection models
    - Fixing batch detection issues
    - Improving overall quality
    """
    # Verify album ownership
    album = db.query(Album).filter(Album.id == album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {album_id} not found"
        )
    
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get photos to reprocess
    photos_query = db.query(Photo).filter(Photo.album_id == album_id)
    
    if not request.force_all and request.min_quality is not None:
        # Find photos with low-quality faces
        faces = db.query(Face).join(Photo).filter(Photo.album_id == album_id).all()
        
        low_quality_photo_ids = set()
        for face in faces:
            blur_score = face.blur_score or 0.0
            brightness_score = face.brightness_score or 0.0
            confidence = face.confidence
            
            overall_quality = (blur_score * 0.4 + brightness_score * 0.4 + confidence * 0.2)
            
            if overall_quality < request.min_quality:
                low_quality_photo_ids.add(face.photo_id)
        
        if not low_quality_photo_ids:
            return ReprocessResponse(
                message="No low-quality faces found",
                face_ids=[]
            )
        
        photos = db.query(Photo).filter(Photo.id.in_(low_quality_photo_ids)).all()
    else:
        photos = photos_query.all()
    
    if not photos:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No photos to reprocess"
        )
    
    # Enqueue reprocessing tasks
    photo_ids = [str(p.id) for p in photos]
    task_id = process_album_photos_task.delay(str(album_id), photo_ids)
    
    logger.info(f"Enqueued album reprocessing task {task_id} for {len(photos)} photos in album {album_id}")
    
    return ReprocessResponse(
        message=f"Album reprocessing started for {len(photos)} photos",
        job_id=str(task_id),
        status_url=f"/api/v1/jobs/{task_id}"
    )





@router.get(
    "/health",
    summary="Face service health check",
    description="Check if face detection pipeline is operational"
)
async def health_check(
    pipeline: FacePipeline = Depends(get_pipeline)
):
    """Health check endpoint for face service."""
    try:
        info = pipeline.get_device_info()
        
        return {
            "status": "healthy",
            "device": info['device'],
            "detector": info['detector'],
            "embedder": info.get('embedder_backend', 'unknown'),
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )




# ============================================================================
# Advanced Clustering Endpoints
# ============================================================================

