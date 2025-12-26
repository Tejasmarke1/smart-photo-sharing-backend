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
    JobAccepted
)
from src.tasks.workers.face_processor import process_faces_task, process_album_photos_task, cluster_album_task
from src.core.cache import cache_result, invalidate_cache
from src.core.rate_limiter import rate_limit
from src.utils.validators import validate_image

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize face pipeline (singleton)
_pipeline: Optional[FacePipeline] = None


def get_pipeline() -> FacePipeline:
    """Get or create face pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = create_pipeline()
        logger.info("Face pipeline initialized")
    return _pipeline


# ============================================================================
# Pydantic Schemas
# ============================================================================


# ============================================================================
# Helper Functions
# ============================================================================

async def get_face_or_404(
    face_id: UUID,
    db: Session,
    current_user: User
) -> Face:
    """Get face by ID or raise 404."""
    face = db.query(Face).filter(Face.id == face_id).first()
    
    if not face:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face {face_id} not found"
        )
    
    # Check permission via photo -> album
    photo = db.query(Photo).filter(Photo.id == face.photo_id).first()
    if not photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated photo not found"
        )
    
    album = db.query(Album).filter(Album.id == photo.album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated album not found"
        )
    
    # Check user has access
    # Support both ORM user and dict for tests
    user_id = getattr(current_user, 'id', None) if current_user is not None else None
    role = getattr(current_user, 'role', None) if current_user is not None else None
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')

    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return face


def serialize_face(face: Face, s3_service: S3Service) -> FaceResponse:
    """Serialize face object to response schema."""
    thumbnail_url = None
    if face.thumbnail_s3_key:
        thumbnail_url = s3_service.generate_presigned_download_url(
            face.thumbnail_s3_key,
            expires_in=3600
        )
    
    person_id = None
    person_name = None
    if face.person_mapping:
        person_id = face.person_mapping.person_id
        person_name = face.person_mapping.person.name if face.person_mapping.person else None
    
    return FaceResponse(
        id=face.id,
        photo_id=face.photo_id,
        bbox=json.loads(face.bbox),
        confidence=face.confidence,
        thumbnail_url=thumbnail_url,
        blur_score=face.blur_score,
        brightness_score=face.brightness_score,
        person_id=person_id,
        person_name=person_name,
        created_at=face.created_at
    )


# ============================================================================
# API Endpoints
# ============================================================================

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
    "/search/by-selfie",
    response_model=List[FaceSearchResponse],
    summary="Search by selfie upload",
    description="Upload a selfie and find matching faces across albums"
)
@rate_limit(key_prefix="search_by_selfie", max_calls=20, period=60)
async def search_by_selfie(
    file: UploadFile = File(..., description="Selfie image (JPG/PNG)"),
    album_id: Optional[UUID] = Query(None, description="Limit to specific album"),
    k: int = Query(50, ge=1, le=200, description="Number of results"),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description="Similarity threshold"),
    min_quality: Optional[float] = Query(None, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Search for matching faces using an uploaded selfie.
    
    Process:
    1. Detect face in selfie
    2. Generate embedding
    3. Search vector database
    4. Return matched faces with scores
    """
    # Validate file
    if file.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are supported"
        )
    
    # Read image
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image too large (max 10MB)"
        )
    
    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        # Search using pipeline
        results = pipeline.search_by_selfie(
            image_rgb,
            album_id=str(album_id) if album_id else None,
            k=k,
            threshold=threshold
        )
        
        if not results:
            return []
        
        # Fetch face details from DB
        face_ids = [UUID(r['face_id']) for r in results]
        faces = db.query(Face).filter(Face.id.in_(face_ids)).all()
        
        # Create lookup
        face_lookup = {str(f.id): f for f in faces}
        
        # Build response
        response = []
        for result in results:
            face_id = result['face_id']
            face = face_lookup.get(face_id)
            
            if not face:
                continue
            
            # Apply quality filter
            if min_quality:
                avg_quality = (face.blur_score + face.brightness_score) / 2 if face.blur_score and face.brightness_score else 0
                if avg_quality < min_quality:
                    continue
            
            # Get thumbnail URL
            thumbnail_url = None
            if face.thumbnail_s3_key:
                thumbnail_url = s3_service.generate_presigned_download_url(
                    face.thumbnail_s3_key,
                    expires_in=3600
                )
            
            # Get person info
            person_id = None
            person_name = None
            if face.person_mapping:
                person_id = face.person_mapping.person_id
                person_name = face.person_mapping.person.name if face.person_mapping.person else None
            
            response.append(FaceSearchResponse(
                face_id=face.id,
                photo_id=face.photo_id,
                similarity_score=result['score'],
                thumbnail_url=thumbnail_url,
                person_id=person_id,
                person_name=person_name,
                bbox=json.loads(face.bbox),
                confidence=face.confidence
            ))
        
        logger.info(f"Selfie search returned {len(response)} results")
        return response
    
    except Exception as e:
        logger.error(f"Selfie search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face search failed: {str(e)}"
        )


@router.post(
    "/search/by-embedding",
    response_model=List[FaceSearchResponse],
    summary="Search by embedding vector",
    description="Search using pre-computed face embedding (privacy-preserving)"
)
@rate_limit(key_prefix="search_by_embedding", max_calls=50, period=60)
async def search_by_embedding(
    request: FaceSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Search for matching faces using a pre-computed embedding.
    
    This endpoint is privacy-preserving:
    - Client computes embedding locally
    - Only vector is transmitted
    - No image data leaves client device
    """
    try:
        # Convert to numpy array
        query_embedding = np.array(request.embedding, dtype=np.float32)
        
        # Normalize
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        results = pipeline.search_engine.search(
            query_embedding,
            k=request.k,
            threshold=request.threshold
        )
        
        if not results:
            return []
        
        # Fetch faces
        face_ids = [UUID(r['face_id']) for r in results]
        
        # Build query with album filter if provided
        query = db.query(Face).filter(Face.id.in_(face_ids))
        
        if request.album_id:
            query = query.join(Photo).filter(Photo.album_id == request.album_id)
        
        faces = query.all()
        
        # Create lookup
        face_lookup = {str(f.id): f for f in faces}
        
        # Build response
        response = []
        for result in results:
            face_id = result['face_id']
            face = face_lookup.get(face_id)
            
            if not face:
                continue
            
            # Apply quality filter
            if request.min_quality:
                avg_quality = (face.blur_score + face.brightness_score) / 2 if face.blur_score and face.brightness_score else 0
                if avg_quality < request.min_quality:
                    continue
            
            # Get thumbnail URL
            thumbnail_url = None
            if face.thumbnail_s3_key:
                thumbnail_url = s3_service.generate_presigned_download_url(
                    face.thumbnail_s3_key,
                    expires_in=3600
                )
            
            # Get person info
            person_id = None
            person_name = None
            if face.person_mapping:
                person_id = face.person_mapping.person_id
                person_name = face.person_mapping.person.name if face.person_mapping.person else None
            
            response.append(FaceSearchResponse(
                face_id=face.id,
                photo_id=face.photo_id,
                similarity_score=result['score'],
                thumbnail_url=thumbnail_url,
                person_id=person_id,
                person_name=person_name,
                bbox=json.loads(face.bbox),
                confidence=face.confidence
            ))
        
        logger.info(f"Embedding search returned {len(response)} results")
        return response
    
    except Exception as e:
        logger.error(f"Embedding search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face search failed: {str(e)}"
        )


@router.post(
    "/albums/{album_id}/cluster",
    response_model=Dict[str, Any],
    summary="Cluster faces in album",
    description="Run clustering to group similar faces (find same people)",
    status_code=status.HTTP_202_ACCEPTED
)
@require_roles(['photographer', 'admin','guest'])
async def cluster_album(
    album_id: UUID,
    request: FaceClusterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Trigger face clustering for an album.
    
    This is an async operation that:
    1. Groups similar faces together
    2. Suggests person clusters
    3. Identifies potential merge candidates
    
    Returns a job ID for status tracking.
    """
    # Verify album ownership
    album = db.query(Album).filter(Album.id == album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {album_id} not found"
        )
    
    if album.photographer_id != current_user.id and current_user.role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Check if album has faces
    face_count = db.query(func.count(Face.id)).join(Photo).filter(
        Photo.album_id == album_id
    ).scalar()
    
    if face_count < request.min_cluster_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Album has only {face_count} faces, need at least {request.min_cluster_size}"
        )
    
    # Enqueue clustering task
    task_id = cluster_album_task.delay(
        str(album_id),
        min_cluster_size=request.min_cluster_size,
        similarity_threshold=request.similarity_threshold,
        merge_threshold=request.merge_threshold
    )
    
    logger.info(f"Enqueued clustering task {task_id} for album {album_id}")
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "message": "Clustering job started",
            "job_id": str(task_id),
            "album_id": str(album_id),
            "face_count": face_count,
            "status_url": f"/api/v1/jobs/{task_id}"
        }
    )


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


class AlbumDetectRequest(BaseModel):
    photo_ids: Optional[List[UUID]] = Field(None, description="Process only these photos; default is all in album")


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


@router.get(
    "/faces/{face_id}/similar",
    response_model=List[FaceSearchResponse],
    summary="Find similar faces",
    description="Find faces similar to a given face"
)
@cache_result(ttl=300)
async def find_similar_faces(
    face_id: UUID,
    k: int = Query(20, ge=1, le=100),
    threshold: float = Query(0.7, ge=0.0, le=1.0),
    same_album_only: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Find faces similar to a given face.
    
    Useful for:
    - Verifying cluster quality
    - Manual labeling assistance
    - Finding duplicates
    """
    face = await get_face_or_404(face_id, db, current_user)
    
    # Check if face has embedding
    if face.embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Face has no embedding"
        )
    
    # Get album ID if filtering
    album_id = None
    if same_album_only:
        photo = db.query(Photo).filter(Photo.id == face.photo_id).first()
        album_id = str(photo.album_id) if photo else None
    
    # Search for similar faces
    query_embedding = np.array(face.embedding, dtype=np.float32)
    
    results = pipeline.search_engine.search(
        query_embedding,
        k=k + 1,  # +1 to exclude self
        threshold=threshold
    )
    
    # Filter out the query face itself
    results = [r for r in results if UUID(r['face_id']) != face_id][:k]
    
    if not results:
        return []
    
    # Fetch faces
    face_ids = [UUID(r['face_id']) for r in results]
    
    query_db = db.query(Face).filter(Face.id.in_(face_ids))
    
    if same_album_only and album_id:
        query_db = query_db.join(Photo).filter(Photo.album_id == UUID(album_id))
    
    faces = query_db.all()
    face_lookup = {str(f.id): f for f in faces}
    
    # Build response
    response = []
    for result in results:
        face_id_str = result['face_id']
        face_obj = face_lookup.get(face_id_str)
        
        if not face_obj:
            continue
        
        thumbnail_url = None
        if face_obj.thumbnail_s3_key:
            thumbnail_url = s3_service.generate_presigned_url(
                face_obj.thumbnail_s3_key,
                expiration=3600
            )
        
        person_id = None
        person_name = None
        if face_obj.person_mapping:
            person_id = face_obj.person_mapping.person_id
            person_name = face_obj.person_mapping.person.name if face_obj.person_mapping.person else None
        
        response.append(FaceSearchResponse(
            face_id=face_obj.id,
            photo_id=face_obj.photo_id,
            similarity_score=result['score'],
            thumbnail_url=thumbnail_url,
            person_id=person_id,
            person_name=person_name,
            bbox=eval(face_obj.bbox),
            confidence=face_obj.confidence
        ))
    
    return response


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
# Error Handlers
# ============================================================================


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