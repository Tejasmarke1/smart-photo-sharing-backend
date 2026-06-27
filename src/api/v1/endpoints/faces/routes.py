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

@router.get(
    "/albums/{album_id}/faces",
    response_model=FaceListResult,
    summary="List faces in album",
    description="Get all detected faces in an album with optional quality filtering"
)
@rate_limit(key_prefix="list_album_faces", max_calls=100, period=60)
async def list_album_faces(
    album_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    min_blur_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    min_brightness_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    person_id: Optional[UUID] = Query(None, description="Filter by person"),
    unlabeled_only: bool = Query(False, description="Show only unlabeled faces"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    List all faces in an album with filtering options.
    
    Supports:
    - Pagination
    - Quality filtering (blur, brightness, confidence)
    - Person filtering
    - Unlabeled faces only
    """
    # Verify album access
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
    
    # Build query
    query = db.query(Face).join(Photo).filter(Photo.album_id == album_id)
    
    # Apply quality filters from query params
    if min_blur_score is not None:
        query = query.filter(Face.blur_score >= min_blur_score)
    if min_brightness_score is not None:
        query = query.filter(Face.brightness_score >= min_brightness_score)
    if min_confidence is not None:
        query = query.filter(Face.confidence >= min_confidence)
    
    # Filter by person
    if person_id:
        query = query.join(FacePerson).filter(FacePerson.person_id == person_id)
    
    # Filter unlabeled
    if unlabeled_only:
        query = query.outerjoin(FacePerson).filter(FacePerson.face_id.is_(None))
    
    # Get total count
    total = query.count()
    
    # Apply pagination and fetch
    faces = query.order_by(Face.created_at.desc()).offset(skip).limit(limit).all()
    
    # Serialize
    response = [serialize_face(face, s3_service) for face in faces]
    
    return {
        "faces": [r.model_dump(mode='json') for r in response],
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.get(
    "/faces/{face_id}",
    response_model=FaceResponse,
    summary="Get face details",
    description="Retrieve detailed information about a specific face"
)
async def get_face(
    face_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """Get detailed information about a face."""
    face = await get_face_or_404(face_id, db, current_user)
    return serialize_face(face, s3_service)


@router.post(
    "/faces/{face_id}/label",
    response_model=FaceResponse,
    summary="Label a face",
    description="Associate a face with a person (create new or link existing)"
)
@rate_limit(key_prefix="label_face", max_calls=50, period=60)
async def label_face(
    face_id: UUID,
    request: FaceLabelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Label a face with a person.
    
    Can either:
    1. Link to existing person (provide person_id)
    2. Create new person (provide person_name)
    """
    face = await get_face_or_404(face_id, db, current_user)
    
    # Get album for permission check
    photo = db.query(Photo).filter(Photo.id == face.photo_id).first()
    album = db.query(Album).filter(Album.id == photo.album_id).first()
    
    # Determine person
    person = None
    if request.person_id:
        person = db.query(Person).filter(
            and_(
                Person.id == request.person_id,
                Person.album_id == album.id
            )
        ).first()
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {request.person_id} not found in this album"
            )
    
    elif request.person_name:
        # Check if person with this name already exists in album
        person = db.query(Person).filter(
            and_(
                Person.album_id == album.id,
                func.lower(Person.name) == request.person_name.lower()
            )
        ).first()
        
        if not person:
            # Create new person
            person = Person(
                album_id=album.id,
                name=request.person_name
            )
            db.add(person)
            db.flush()
            logger.info(f"Created new person {person.id} in album {album.id}")
    
    # Check if face is already labeled
    existing_mapping = db.query(FacePerson).filter(
        FacePerson.face_id == face_id
    ).first()
    
    if existing_mapping:
        # Update existing mapping
        existing_mapping.person_id = person.id
        existing_mapping.is_manual = request.is_manual
        existing_mapping.confidence = 1.0 if request.is_manual else None
        logger.info(f"Updated face {face_id} label to person {person.id}")
    else:
        # Create new mapping
        mapping = FacePerson(
            face_id=face_id,
            person_id=person.id,
            is_manual=request.is_manual,
            confidence=1.0 if request.is_manual else None
        )
        db.add(mapping)
        logger.info(f"Labeled face {face_id} as person {person.id}")
    
    db.commit()
    db.refresh(face)
    
    # Invalidate cache
    await invalidate_cache(f"album_faces_{album.id}")
    
    return serialize_face(face, s3_service)


@router.delete(
    "/faces/{face_id}/label",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove face label",
    description="Remove person association from a face"
)
async def unlabel_face(
    face_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Remove person label from a face."""
    face = await get_face_or_404(face_id, db, current_user)
    
    # Delete mapping
    db.query(FacePerson).filter(FacePerson.face_id == face_id).delete()
    db.commit()
    
    logger.info(f"Removed label from face {face_id}")
    
    # Invalidate cache
    photo = db.query(Photo).filter(Photo.id == face.photo_id).first()
    await invalidate_cache(f"album_faces_{photo.album_id}")
    
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content={})


@router.post(
    "/faces/batch-delete",
    response_model=Dict[str, Any],
    summary="Batch delete faces",
    description="Delete multiple faces and optionally their thumbnails"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="batch_delete_faces", max_calls=10, period=60)
async def batch_delete_faces(
    request: BatchDeleteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Delete multiple faces in one operation.
    
    Features:
    - Bulk deletion
    - Optional thumbnail cleanup
    - Cascade delete mappings
    
    Use cases:
    - Remove low-quality detections
    - Clean up after reprocessing
    - Delete mislabeled faces
    """
    # Fetch faces
    faces = db.query(Face).filter(Face.id.in_(request.face_ids)).all()
    
    if not faces:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No faces found"
        )
    
    # Verify permissions
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
    
    accessible_faces = []
    for face in faces:
        photo = db.query(Photo).filter(Photo.id == face.photo_id).first()
        if photo:
            album = db.query(Album).filter(Album.id == photo.album_id).first()
            if album and (album.photographer_id == user_id or role == 'admin'):
                accessible_faces.append(face)
    
    if not accessible_faces:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No accessible faces to delete"
        )
    
    # Delete thumbnails from S3 if requested
    deleted_thumbnails = 0
    if request.delete_thumbnails:
        for face in accessible_faces:
            if face.thumbnail_s3_key:
                try:
                    s3_service.delete_file(face.thumbnail_s3_key)
                    deleted_thumbnails += 1
                except Exception as e:
                    logger.warning(f"Failed to delete thumbnail {face.thumbnail_s3_key}: {e}")
    
    # Delete face mappings (cascade)
    db.query(FacePerson).filter(
        FacePerson.face_id.in_([f.id for f in accessible_faces])
    ).delete(synchronize_session=False)
    
    # Delete faces
    face_ids_to_delete = [f.id for f in accessible_faces]
    db.query(Face).filter(Face.id.in_(face_ids_to_delete)).delete(synchronize_session=False)
    
    db.commit()
    
    logger.info(f"Batch deleted {len(accessible_faces)} faces (thumbnails: {deleted_thumbnails})")
    
    return {
        "message": "Faces deleted successfully",
        "deleted_count": len(accessible_faces),
        "deleted_thumbnails": deleted_thumbnails,
        "face_ids": [str(f.id) for f in accessible_faces]
    }


async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )



async def general_error_handler(request, exc):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )