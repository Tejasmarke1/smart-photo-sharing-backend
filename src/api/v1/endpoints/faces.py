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


# ============================================================================
# Quality Assessment Endpoints
# ============================================================================




@router.get(
    "/faces/{face_id}/quality",
    response_model=FaceQualityResponse,
    summary="Get face quality assessment",
    description="Detailed quality metrics for a single face"
)
@rate_limit(key_prefix="face_quality", max_calls=100, period=60)
async def get_face_quality(
    face_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Get detailed quality assessment for a face.
    
    Quality metrics include:
    - Blur score (0-1, higher is better)
    - Brightness score (0-1, higher is better)
    - Detection confidence (0-1)
    - Overall quality grade (A-F)
    """
    face = await get_face_or_404(face_id, db, current_user)
    
    # Calculate overall quality
    blur_score = face.blur_score or 0.0
    brightness_score = face.brightness_score or 0.0
    confidence = face.confidence
    
    # Weighted average: blur and brightness more important than confidence
    overall_quality = (blur_score * 0.4 + brightness_score * 0.4 + confidence * 0.2)
    
    # Grade mapping
    if overall_quality >= 0.85:
        grade = "A"
    elif overall_quality >= 0.70:
        grade = "B"
    elif overall_quality >= 0.55:
        grade = "C"
    elif overall_quality >= 0.40:
        grade = "D"
    else:
        grade = "F"
    
    # Identify issues
    issues = []
    if blur_score < 0.5:
        issues.append("Image is blurry")
    if brightness_score < 0.4:
        issues.append("Poor lighting/exposure")
    if brightness_score > 0.9:
        issues.append("Overexposed")
    if confidence < 0.7:
        issues.append("Low detection confidence")
    
    # Get thumbnail URL
    thumbnail_url = None
    if face.thumbnail_s3_key:
        thumbnail_url = s3_service.generate_presigned_download_url(
            face.thumbnail_s3_key,
            expires_in=3600
        )
    
    return FaceQualityResponse(
        face_id=face.id,
        photo_id=face.photo_id,
        blur_score=blur_score,
        brightness_score=brightness_score,
        confidence=confidence,
        overall_quality=overall_quality,
        quality_grade=grade,
        issues=issues,
        thumbnail_url=thumbnail_url
    )


@router.post(
    "/faces/quality-check",
    response_model=List[FaceQualityResponse],
    summary="Batch quality check",
    description="Check quality for multiple faces at once"
)
@rate_limit(key_prefix="batch_quality_check", max_calls=20, period=60)
async def batch_quality_check(
    request: QualityCheckRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Perform quality assessment on multiple faces.
    
    Useful for:
    - Identifying low-quality faces for deletion
    - Batch quality filtering
    - Quality reports
    """
    # Fetch all requested faces
    faces = db.query(Face).filter(Face.id.in_(request.face_ids)).all()
    
    if not faces:
        return []
    
    # Verify access to at least one face
    face_ids_set = {f.id for f in faces}
    accessible_ids = set()
    
    for face in faces:
        photo = db.query(Photo).filter(Photo.id == face.photo_id).first()
        if photo:
            album = db.query(Album).filter(Album.id == photo.album_id).first()
            user_id = getattr(current_user, 'id', None)
            role = getattr(current_user, 'role', None)
            if isinstance(current_user, dict):
                user_id = user_id or current_user.get('id')
                role = role or current_user.get('role')
            
            if album and (album.photographer_id == user_id or role == 'admin'):
                accessible_ids.add(face.id)
    
    # Filter to only accessible faces
    faces = [f for f in faces if f.id in accessible_ids]
    
    if not faces:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No accessible faces in the provided list"
        )
    
    # Thresholds based on mode
    blur_threshold = 0.6 if request.strict_mode else 0.5
    brightness_low = 0.5 if request.strict_mode else 0.4
    brightness_high = 0.85 if request.strict_mode else 0.9
    confidence_threshold = 0.8 if request.strict_mode else 0.7
    
    # Assess each face
    results = []
    for face in faces:
        blur_score = face.blur_score or 0.0
        brightness_score = face.brightness_score or 0.0
        confidence = face.confidence
        
        overall_quality = (blur_score * 0.4 + brightness_score * 0.4 + confidence * 0.2)
        
        if overall_quality >= 0.85:
            grade = "A"
        elif overall_quality >= 0.70:
            grade = "B"
        elif overall_quality >= 0.55:
            grade = "C"
        elif overall_quality >= 0.40:
            grade = "D"
        else:
            grade = "F"
        
        issues = []
        if blur_score < blur_threshold:
            issues.append("Image is blurry")
        if brightness_score < brightness_low:
            issues.append("Poor lighting/exposure")
        if brightness_score > brightness_high:
            issues.append("Overexposed")
        if confidence < confidence_threshold:
            issues.append("Low detection confidence")
        
        thumbnail_url = None
        if face.thumbnail_s3_key:
            thumbnail_url = s3_service.generate_presigned_download_url(
                face.thumbnail_s3_key,
                expires_in=3600
            )
        
        results.append(FaceQualityResponse(
            face_id=face.id,
            photo_id=face.photo_id,
            blur_score=blur_score,
            brightness_score=brightness_score,
            confidence=confidence,
            overall_quality=overall_quality,
            quality_grade=grade,
            issues=issues,
            thumbnail_url=thumbnail_url
        ))
    
    return results


@router.get(
    "/albums/{album_id}/faces/low-quality",
    response_model=List[FaceQualityResponse],
    summary="Find low-quality faces in album",
    description="Identify faces with quality issues for review or deletion"
)
@rate_limit(key_prefix="low_quality_faces", max_calls=50, period=60)
async def get_low_quality_faces(
    album_id: UUID,
    min_overall_quality: float = Query(0.5, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Find faces in an album that fall below quality thresholds.
    
    Use cases:
    - Clean up low-quality detections
    - Review problematic faces
    - Identify photos that need reprocessing
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
    
    # Query faces with calculated quality
    faces = db.query(Face).join(Photo).filter(
        Photo.album_id == album_id
    ).all()
    
    # Calculate quality and filter
    low_quality_faces = []
    for face in faces:
        blur_score = face.blur_score or 0.0
        brightness_score = face.brightness_score or 0.0
        confidence = face.confidence
        
        overall_quality = (blur_score * 0.4 + brightness_score * 0.4 + confidence * 0.2)
        
        if overall_quality < min_overall_quality:
            if overall_quality >= 0.85:
                grade = "A"
            elif overall_quality >= 0.70:
                grade = "B"
            elif overall_quality >= 0.55:
                grade = "C"
            elif overall_quality >= 0.40:
                grade = "D"
            else:
                grade = "F"
            
            issues = []
            if blur_score < 0.5:
                issues.append("Image is blurry")
            if brightness_score < 0.4:
                issues.append("Poor lighting/exposure")
            if brightness_score > 0.9:
                issues.append("Overexposed")
            if confidence < 0.7:
                issues.append("Low detection confidence")
            
            thumbnail_url = None
            if face.thumbnail_s3_key:
                thumbnail_url = s3_service.generate_presigned_download_url(
                    face.thumbnail_s3_key,
                    expires_in=3600
                )
            
            low_quality_faces.append(FaceQualityResponse(
                face_id=face.id,
                photo_id=face.photo_id,
                blur_score=blur_score,
                brightness_score=brightness_score,
                confidence=confidence,
                overall_quality=overall_quality,
                quality_grade=grade,
                issues=issues,
                thumbnail_url=thumbnail_url
            ))
    
    # Sort by quality (worst first) and limit
    low_quality_faces.sort(key=lambda x: x.overall_quality)
    return low_quality_faces[:limit]


# ============================================================================
# Face Filtering Endpoints
# ============================================================================




@router.post(
    "/faces/filter",
    response_model=List[FaceResponse],
    summary="Advanced face filtering",
    description="Filter faces using complex criteria"
)
@rate_limit(key_prefix="filter_faces", max_calls=50, period=60)
async def filter_faces(
    request: AdvancedFilterRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Advanced filtering for faces across multiple dimensions.
    
    Supports:
    - Album filtering
    - Person filtering
    - Quality thresholds
    - Date ranges
    - Labeling status
    """
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    
    # Build base query
    query = db.query(Face).join(Photo).join(Album)
    
    # Permission filter: only user's albums unless admin
    if role != 'admin':
        query = query.filter(Album.photographer_id == user_id)
    
    # Album filter
    if request.album_ids:
        query = query.filter(Photo.album_id.in_(request.album_ids))
    
    # Confidence filters
    if request.min_confidence is not None:
        query = query.filter(Face.confidence >= request.min_confidence)
    if request.max_confidence is not None:
        query = query.filter(Face.confidence <= request.max_confidence)
    
    # Quality filters
    if request.min_blur_score is not None:
        query = query.filter(Face.blur_score >= request.min_blur_score)
    if request.min_brightness_score is not None:
        query = query.filter(Face.brightness_score >= request.min_brightness_score)
    
    # Person filters
    if request.person_ids:
        query = query.join(FacePerson).filter(FacePerson.person_id.in_(request.person_ids))
    
    if request.unlabeled_only:
        query = query.outerjoin(FacePerson).filter(FacePerson.face_id.is_(None))
    
    # Thumbnail filter
    if request.has_thumbnail is not None:
        if request.has_thumbnail:
            query = query.filter(Face.thumbnail_s3_key.isnot(None))
        else:
            query = query.filter(Face.thumbnail_s3_key.is_(None))
    
    # Date filters
    if request.created_after:
        try:
            after_dt = datetime.fromisoformat(request.created_after.replace('Z', '+00:00'))
            query = query.filter(Face.created_at >= after_dt)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid created_after date format"
            )
    
    if request.created_before:
        try:
            before_dt = datetime.fromisoformat(request.created_before.replace('Z', '+00:00'))
            query = query.filter(Face.created_at <= before_dt)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid created_before date format"
            )
    
    # Execute query
    faces = query.order_by(Face.created_at.desc()).limit(request.limit).all()
    
    # Serialize
    return [serialize_face(face, s3_service) for face in faces]





@router.get(
    "/faces/duplicates",
    response_model=List[DuplicateFace],
    summary="Find duplicate faces",
    description="Detect near-duplicate face detections (same person, different photos)"
)
@rate_limit(key_prefix="find_duplicates", max_calls=20, period=60)
@cache_result(ttl=600)
async def find_duplicate_faces(
    album_id: Optional[UUID] = Query(None, description="Limit to specific album"),
    threshold: float = Query(0.95, ge=0.8, le=0.99),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Find near-duplicate faces (very high similarity).
    
    Use cases:
    - Identify redundant face detections
    - Clean up duplicate entries
    - Merge similar faces
    """
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    
    # Build query
    query = db.query(Face).join(Photo).join(Album)
    
    if role != 'admin':
        query = query.filter(Album.photographer_id == user_id)
    
    if album_id:
        query = query.filter(Photo.album_id == album_id)
    
    # Only faces with embeddings
    query = query.filter(Face.embedding.isnot(None))
    
    faces = query.all()
    
    if len(faces) < 2:
        return []
    
    # Find pairs with high similarity
    duplicates = []
    processed_pairs = set()
    
    for i, face1 in enumerate(faces):
        if len(duplicates) >= limit:
            break
        
        emb1 = np.array(face1.embedding, dtype=np.float32)
        
        # Search for similar faces
        results = pipeline.search_engine.search(emb1, k=10, threshold=threshold)
        
        for result in results:
            face2_id = UUID(result['face_id'])
            
            # Skip self
            if face2_id == face1.id:
                continue
            
            # Skip if already processed
            pair = tuple(sorted([str(face1.id), str(face2_id)]))
            if pair in processed_pairs:
                continue
            
            processed_pairs.add(pair)
            
            # Find face2
            face2 = next((f for f in faces if f.id == face2_id), None)
            if not face2:
                continue
            
            # Skip if same photo
            if face1.photo_id == face2.photo_id:
                continue
            
            # Get thumbnails
            thumb1 = None
            if face1.thumbnail_s3_key:
                thumb1 = s3_service.generate_presigned_download_url(
                    face1.thumbnail_s3_key, expires_in=3600
                )
            
            thumb2 = None
            if face2.thumbnail_s3_key:
                thumb2 = s3_service.generate_presigned_download_url(
                    face2.thumbnail_s3_key, expires_in=3600
                )
            
            duplicates.append(DuplicateFace(
                face_id_1=face1.id,
                face_id_2=face2.id,
                similarity_score=result['score'],
                photo_id_1=face1.photo_id,
                photo_id_2=face2.photo_id,
                thumbnail_url_1=thumb1,
                thumbnail_url_2=thumb2
            ))
            
            if len(duplicates) >= limit:
                break
    
    return duplicates





@router.get(
    "/faces/outliers",
    response_model=List[OutlierFace],
    summary="Find face outliers",
    description="Detect faces that may be mislabeled (low similarity to their person cluster)"
)
@rate_limit(key_prefix="find_outliers", max_calls=20, period=60)
async def find_face_outliers(
    person_id: Optional[UUID] = Query(None, description="Check specific person"),
    threshold: float = Query(0.6, ge=0.3, le=0.8, description="Similarity threshold"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Find faces that don't fit their assigned person.
    
    Helps identify:
    - Mislabeled faces
    - Incorrect person assignments
    - Faces needing re-review
    """
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    
    # Get persons to check
    person_query = db.query(Person)
    if role != 'admin':
        person_query = person_query.filter(Person.photographer_id == user_id)
    
    if person_id:
        person_query = person_query.filter(Person.id == person_id)
    
    persons = person_query.all()
    
    outliers = []
    
    for person in persons:
        if len(outliers) >= limit:
            break
        
        # Get all faces for this person
        face_mappings = db.query(FacePerson).filter(
            FacePerson.person_id == person.id
        ).all()
        
        if len(face_mappings) < 2:
            continue  # Need at least 2 faces to compare
        
        face_ids = [fm.face_id for fm in face_mappings]
        faces = db.query(Face).filter(Face.id.in_(face_ids)).all()
        
        # Filter faces with embeddings
        faces_with_emb = [f for f in faces if f.embedding is not None]
        
        if len(faces_with_emb) < 2:
            continue
        
        # Calculate average embedding for the cluster
        embeddings = np.array([f.embedding for f in faces_with_emb], dtype=np.float32)
        
        # For each face, calculate similarity to others
        for face in faces_with_emb:
            emb = np.array(face.embedding, dtype=np.float32)
            
            # Calculate cosine similarity to all other faces
            other_embs = embeddings[[i for i, f in enumerate(faces_with_emb) if f.id != face.id]]
            
            if len(other_embs) == 0:
                continue
            
            # Cosine similarity
            similarities = np.dot(other_embs, emb) / (
                np.linalg.norm(other_embs, axis=1) * np.linalg.norm(emb)
            )
            
            avg_similarity = float(np.mean(similarities))
            
            # If average similarity is below threshold, it's an outlier
            if avg_similarity < threshold:
                thumbnail_url = None
                if face.thumbnail_s3_key:
                    thumbnail_url = s3_service.generate_presigned_download_url(
                        face.thumbnail_s3_key, expires_in=3600
                    )
                
                outliers.append(OutlierFace(
                    face_id=face.id,
                    photo_id=face.photo_id,
                    person_id=person.id,
                    person_name=person.name,
                    confidence=face.confidence,
                    avg_similarity_to_cluster=avg_similarity,
                    thumbnail_url=thumbnail_url
                ))
                
                if len(outliers) >= limit:
                    break
    
    # Sort by similarity (lowest first)
    outliers.sort(key=lambda x: x.avg_similarity_to_cluster)
    return outliers[:limit]


# ============================================================================
# Face Reprocessing Endpoints
# ============================================================================



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

@router.post(
    "/albums/{album_id}/cluster/auto",
    response_model=ClusterStatusResponse,
    summary="Auto-cluster with smart defaults",
    description="Automatically cluster faces using intelligent algorithm parameters",
    status_code=status.HTTP_202_ACCEPTED
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="auto_cluster", max_calls=5, period=300)
async def auto_cluster_album(
    album_id: UUID,
    request: AutoClusterRequest = Body(default_factory=AutoClusterRequest),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline)
):
    """
    Auto-cluster faces with smart defaults.
    
    Smart defaults:
    - Automatically determine optimal cluster size
    - Adaptive similarity threshold based on data
    - Intelligent noise filtering
    
    Returns job ID for status tracking.
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
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Count faces
    face_count = db.query(func.count(Face.id)).join(Photo).filter(
        Photo.album_id == album_id,
        Face.embedding.isnot(None)
    ).scalar()
    
    if face_count < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Album has only {face_count} faces with embeddings, need at least 3"
        )
    
    # Determine parameters
    if request.use_smart_defaults:
        # Smart parameter selection based on dataset size
        if face_count < 20:
            min_cluster_size = 2
            similarity_threshold = 0.65
        elif face_count < 100:
            min_cluster_size = 3
            similarity_threshold = 0.70
        else:
            min_cluster_size = 5
            similarity_threshold = 0.75
    else:
        min_cluster_size = request.min_cluster_size or 3
        similarity_threshold = request.similarity_threshold or 0.70
    
    # Enqueue clustering task
    from src.tasks.workers.face_processor import cluster_album_task
    task_id = cluster_album_task.delay(
        str(album_id),
        min_cluster_size=min_cluster_size,
        similarity_threshold=similarity_threshold,
        merge_threshold=0.85
    )
    
    logger.info(f"Enqueued auto-clustering task {task_id} for album {album_id} ({face_count} faces)")
    
    return ClusterStatusResponse(
        job_id=str(task_id),
        status="pending",
        total_faces=face_count,
        started_at=datetime.utcnow()
    )


@router.get(
    "/albums/{album_id}/cluster/status",
    response_model=ClusterStatusResponse,
    summary="Get clustering job status",
    description="Check the status of a clustering job"
)
async def get_clustering_status(
    album_id: UUID,
    job_id: str = Query(..., description="Clustering job ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get clustering job status.
    
    Status values:
    - pending: Job queued
    - running: Processing
    - completed: Done
    - failed: Error occurred
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
    
    # Check Celery task status
    from celery.result import AsyncResult
    task = AsyncResult(job_id)
    
    # Get cluster count if completed
    num_clusters = None
    if task.state == 'SUCCESS':
        clusters = db.query(FaceCluster).filter(
            FaceCluster.album_id == album_id,
            FaceCluster.job_id == job_id
        ).count()
        num_clusters = clusters
    
    return ClusterStatusResponse(
        job_id=job_id,
        status=task.state.lower(),
        progress=None,  # Could add progress tracking
        num_clusters=num_clusters,
        error=str(task.info) if task.state == 'FAILURE' else None
    )


@router.get(
    "/albums/{album_id}/cluster/results",
    response_model=ClusterResultsResponse,
    summary="Get clustering results",
    description="Retrieve complete clustering results for review"
)
@cache_result(ttl=300)
async def get_clustering_results(
    album_id: UUID,
    job_id: Optional[str] = Query(None, description="Specific job ID, or latest if not provided"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Get clustering results for review.
    
    Returns:
    - All clusters with sample faces
    - Noise faces (not clustered)
    - Merge suggestions
    - Cluster quality metrics
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
    
    # Get clusters
    query = db.query(FaceCluster).filter(FaceCluster.album_id == album_id)
    if job_id:
        query = query.filter(FaceCluster.job_id == job_id)
    
    clusters = query.order_by(FaceCluster.size.desc()).all()
    
    if not clusters:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No clustering results found"
        )
    
    # Build cluster details
    cluster_details = []
    all_clustered_face_ids = set()
    
    for cluster in clusters:
        face_ids = cluster.face_ids if isinstance(cluster.face_ids, list) else []
        all_clustered_face_ids.update(face_ids)
        
        # Get sample thumbnails (first 6)
        sample_faces = db.query(Face).filter(Face.id.in_(face_ids[:6])).all()
        sample_thumbnails = []
        
        for face in sample_faces:
            if face.thumbnail_s3_key:
                url = s3_service.generate_presigned_download_url(
                    face.thumbnail_s3_key, expires_in=3600
                )
                sample_thumbnails.append(url)
        
        # Get representative thumbnail
        rep_thumbnail = None
        if cluster.representative_face_id:
            rep_face = db.query(Face).filter(Face.id == cluster.representative_face_id).first()
            if rep_face and rep_face.thumbnail_s3_key:
                rep_thumbnail = s3_service.generate_presigned_download_url(
                    rep_face.thumbnail_s3_key, expires_in=3600
                )
        
        cluster_details.append(ClusterDetailResponse(
            id=cluster.id,
            album_id=cluster.album_id,
            cluster_label=cluster.cluster_label,
            size=cluster.size,
            status=cluster.status,
            avg_similarity=cluster.avg_similarity,
            confidence_score=cluster.confidence_score,
            representative_face_id=cluster.representative_face_id,
            representative_thumbnail_url=rep_thumbnail,
            face_ids=[UUID(fid) for fid in face_ids],
            sample_thumbnails=sample_thumbnails,
            person_id=cluster.person_id,
            person_name=cluster.person.name if cluster.person else None,
            created_at=cluster.created_at
        ))
    
    # Find noise faces (not in any cluster)
    total_faces = db.query(func.count(Face.id)).join(Photo).filter(
        Photo.album_id == album_id,
        Face.embedding.isnot(None)
    ).scalar()
    
    all_face_ids = db.query(Face.id).join(Photo).filter(
        Photo.album_id == album_id,
        Face.embedding.isnot(None)
    ).all()
    
    noise_faces = [fid[0] for fid in all_face_ids if str(fid[0]) not in all_clustered_face_ids]
    
    # Get merge suggestions (clusters with high similarity)
    merge_suggestions = []
    # TODO: Implement merge suggestion algorithm
    
    return ClusterResultsResponse(
        job_id=clusters[0].job_id or "unknown",
        album_id=album_id,
        total_faces=total_faces,
        num_clusters=len(clusters),
        clusters=cluster_details,
        noise_faces=noise_faces,
        merge_suggestions=merge_suggestions,
        completed_at=clusters[0].created_at
    )


@router.post(
    "/albums/{album_id}/cluster/review",
    response_model=Dict[str, Any],
    summary="Submit cluster review",
    description="Review and apply corrections to clustering results"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="cluster_review", max_calls=20, period=60)
async def submit_cluster_review(
    album_id: UUID,
    request: ClusterReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submit cluster review with corrections.
    
    Actions:
    - accept: Convert cluster to person
    - reject: Mark as noise
    - split: Divide into multiple clusters
    - merge: Combine with other clusters
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
    
    results = {
        'accepted': 0,
        'rejected': 0,
        'split': 0,
        'merged': 0,
        'errors': []
    }
    
    for review in request.reviews:
        cluster_id = UUID(review['cluster_id'])
        action = review['action']
        data = review.get('data', {})
        
        cluster = db.query(FaceCluster).filter(
            FaceCluster.id == cluster_id,
            FaceCluster.album_id == album_id
        ).first()
        
        if not cluster:
            results['errors'].append(f"Cluster {cluster_id} not found")
            continue
        
        try:
            if action == 'accept':
                # Create person from cluster
                person_name = data.get('person_name', f'Person {cluster.cluster_label}')
                person = Person(
                    album_id=album_id,
                    name=person_name,
                    representative_face_id=cluster.representative_face_id
                )
                db.add(person)
                db.flush()
                
                # Link faces to person
                face_ids = cluster.face_ids if isinstance(cluster.face_ids, list) else []
                for face_id_str in face_ids:
                    mapping = FacePerson(
                        face_id=UUID(face_id_str),
                        person_id=person.id,
                        is_manual=False
                    )
                    db.add(mapping)
                
                cluster.status = 'accepted'
                cluster.person_id = person.id
                results['accepted'] += 1
                
            elif action == 'reject':
                cluster.status = 'rejected'
                results['rejected'] += 1
                
            elif action == 'split':
                # Mark for split (actual split handled separately)
                cluster.status = 'split'
                results['split'] += 1
                
            elif action == 'merge':
                # Mark for merge (actual merge handled separately)
                cluster.status = 'merged'
                results['merged'] += 1
            
            cluster.reviewed_by_user_id = user_id
            cluster.review_notes = data.get('notes')
            
        except Exception as e:
            logger.error(f"Error processing review for cluster {cluster_id}: {e}")
            results['errors'].append(f"Cluster {cluster_id}: {str(e)}")
    
    db.commit()
    
    return {
        "message": "Review submitted successfully",
        "results": results
    }


# ============================================================================
# Cluster Operations Endpoints
# ============================================================================

@router.get(
    "/clusters/{cluster_id}",
    response_model=ClusterDetailResponse,
    summary="Get cluster details",
    description="Retrieve detailed information about a specific cluster"
)
async def get_cluster_details(
    cluster_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """Get detailed information about a cluster."""
    cluster = db.query(FaceCluster).filter(FaceCluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster {cluster_id} not found"
        )
    
    # Verify album access
    album = db.query(Album).filter(Album.id == cluster.album_id).first()
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
    
    # Get sample thumbnails
    face_ids = cluster.face_ids if isinstance(cluster.face_ids, list) else []
    sample_faces = db.query(Face).filter(Face.id.in_(face_ids[:6])).all()
    sample_thumbnails = []
    
    for face in sample_faces:
        if face.thumbnail_s3_key:
            url = s3_service.generate_presigned_download_url(
                face.thumbnail_s3_key, expires_in=3600
            )
            sample_thumbnails.append(url)
    
    # Get representative thumbnail
    rep_thumbnail = None
    if cluster.representative_face_id:
        rep_face = db.query(Face).filter(Face.id == cluster.representative_face_id).first()
        if rep_face and rep_face.thumbnail_s3_key:
            rep_thumbnail = s3_service.generate_presigned_download_url(
                rep_face.thumbnail_s3_key, expires_in=3600
            )
    
    return ClusterDetailResponse(
        id=cluster.id,
        album_id=cluster.album_id,
        cluster_label=cluster.cluster_label,
        size=cluster.size,
        status=cluster.status,
        avg_similarity=cluster.avg_similarity,
        confidence_score=cluster.confidence_score,
        representative_face_id=cluster.representative_face_id,
        representative_thumbnail_url=rep_thumbnail,
        face_ids=[UUID(fid) for fid in face_ids],
        sample_thumbnails=sample_thumbnails,
        person_id=cluster.person_id,
        person_name=cluster.person.name if cluster.person else None,
        created_at=cluster.created_at
    )


@router.post(
    "/clusters/{cluster_id}/accept",
    response_model=Dict[str, Any],
    summary="Accept cluster as person",
    description="Convert cluster to a person record"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="accept_cluster", max_calls=50, period=60)
async def accept_cluster(
    cluster_id: UUID,
    request: ClusterAcceptRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Accept cluster and create person.
    
    Process:
    1. Create person record
    2. Link all faces to person
    3. Mark cluster as accepted
    """
    cluster = db.query(FaceCluster).filter(FaceCluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster {cluster_id} not found"
        )
    
    # Verify album access
    album = db.query(Album).filter(Album.id == cluster.album_id).first()
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
    
    if cluster.status == 'accepted':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cluster already accepted"
        )
    
    # Create person
    person = Person(
        album_id=cluster.album_id,
        name=request.person_name,
        email=request.person_email,
        phone=request.person_phone,
        representative_face_id=cluster.representative_face_id
    )
    db.add(person)
    db.flush()
    
    # Link faces to person
    face_ids = cluster.face_ids if isinstance(cluster.face_ids, list) else []
    linked_count = 0
    
    for face_id_str in face_ids:
        # Check if already linked
        existing = db.query(FacePerson).filter(
            FacePerson.face_id == UUID(face_id_str)
        ).first()
        
        if not existing:
            mapping = FacePerson(
                face_id=UUID(face_id_str),
                person_id=person.id,
                is_manual=False
            )
            db.add(mapping)
            linked_count += 1
    
    # Update cluster
    cluster.status = 'accepted'
    cluster.person_id = person.id
    cluster.reviewed_by_user_id = user_id
    
    db.commit()
    
    logger.info(f"Accepted cluster {cluster_id} as person {person.id} ({linked_count} faces)")
    
    return {
        "message": "Cluster accepted successfully",
        "person_id": str(person.id),
        "person_name": person.name,
        "faces_linked": linked_count
    }


@router.post(
    "/clusters/{cluster_id}/reject",
    response_model=Dict[str, Any],
    summary="Reject cluster",
    description="Mark cluster as noise/invalid"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="reject_cluster", max_calls=50, period=60)
async def reject_cluster(
    cluster_id: UUID,
    notes: Optional[str] = Body(None, description="Rejection reason"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Reject cluster as noise/invalid.
    
    Rejected clusters won't be suggested again.
    Faces remain available for other operations.
    """
    cluster = db.query(FaceCluster).filter(FaceCluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster {cluster_id} not found"
        )
    
    # Verify album access
    album = db.query(Album).filter(Album.id == cluster.album_id).first()
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
    
    cluster.status = 'rejected'
    cluster.reviewed_by_user_id = user_id
    cluster.review_notes = notes
    
    db.commit()
    
    logger.info(f"Rejected cluster {cluster_id}")
    
    return {
        "message": "Cluster rejected successfully",
        "cluster_id": str(cluster_id)
    }


@router.post(
    "/clusters/{cluster_id}/split",
    response_model=Dict[str, Any],
    summary="Split cluster",
    description="Split a cluster into multiple sub-clusters"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="split_cluster", max_calls=20, period=60)
async def split_cluster(
    cluster_id: UUID,
    request: ClusterSplitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Split cluster into multiple clusters.
    
    Useful when algorithm incorrectly grouped different people.
    """
    cluster = db.query(FaceCluster).filter(FaceCluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster {cluster_id} not found"
        )
    
    # Verify album access
    album = db.query(Album).filter(Album.id == cluster.album_id).first()
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
    
    # Validate face groups
    original_face_ids = set(cluster.face_ids if isinstance(cluster.face_ids, list) else [])
    provided_face_ids = set()
    for group in request.face_groups:
        provided_face_ids.update(str(fid) for fid in group)
    
    if provided_face_ids != original_face_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Face groups must contain all faces from original cluster"
        )
    
    # Get max cluster label
    max_label = db.query(func.max(FaceCluster.cluster_label)).filter(
        FaceCluster.album_id == cluster.album_id
    ).scalar() or 0
    
    # Create new clusters
    new_clusters = []
    for i, group in enumerate(request.face_groups):
        new_cluster = FaceCluster(
            album_id=cluster.album_id,
            job_id=cluster.job_id,
            cluster_label=max_label + i + 1,
            size=len(group),
            status='pending',
            face_ids=[str(fid) for fid in group],
            representative_face_id=group[0]  # Use first face as representative
        )
        db.add(new_cluster)
        new_clusters.append(new_cluster)
    
    # Mark original as split
    cluster.status = 'split'
    cluster.reviewed_by_user_id = user_id
    
    db.commit()
    
    logger.info(f"Split cluster {cluster_id} into {len(new_clusters)} clusters")
    
    return {
        "message": "Cluster split successfully",
        "original_cluster_id": str(cluster_id),
        "new_cluster_ids": [str(c.id) for c in new_clusters],
        "split_count": len(new_clusters)
    }


@router.post(
    "/clusters/merge",
    response_model=Dict[str, Any],
    summary="Merge clusters",
    description="Merge multiple clusters into one"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="merge_clusters", max_calls=20, period=60)
async def merge_clusters(
    request: ClusterMergeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Merge multiple clusters into one.
    
    Useful when algorithm split same person into multiple clusters.
    """
    if len(request.cluster_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 clusters to merge"
        )
    
    # Fetch clusters
    clusters = db.query(FaceCluster).filter(
        FaceCluster.id.in_(request.cluster_ids)
    ).all()
    
    if len(clusters) != len(request.cluster_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Some clusters not found"
        )
    
    # Verify all from same album
    album_ids = set(c.album_id for c in clusters)
    if len(album_ids) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All clusters must be from same album"
        )
    
    # Verify album access
    album = db.query(Album).filter(Album.id == list(album_ids)[0]).first()
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
    
    # Collect all face IDs
    all_face_ids = []
    total_size = 0
    for cluster in clusters:
        face_ids = cluster.face_ids if isinstance(cluster.face_ids, list) else []
        all_face_ids.extend(face_ids)
        total_size += cluster.size
    
    # Create merged cluster
    merged_cluster = FaceCluster(
        album_id=album.id,
        job_id=clusters[0].job_id,
        cluster_label=clusters[0].cluster_label,  # Use first cluster's label
        size=total_size,
        status='pending',
        face_ids=all_face_ids,
        representative_face_id=clusters[0].representative_face_id
    )
    db.add(merged_cluster)
    db.flush()
    
    # Mark originals as merged
    for cluster in clusters:
        cluster.status = 'merged'
        cluster.merged_into_cluster_id = merged_cluster.id
        cluster.reviewed_by_user_id = user_id
    
    db.commit()
    
    logger.info(f"Merged {len(clusters)} clusters into {merged_cluster.id}")
    
    return {
        "message": "Clusters merged successfully",
        "merged_cluster_id": str(merged_cluster.id),
        "original_cluster_ids": [str(c.id) for c in clusters],
        "total_faces": total_size
    }


# ============================================================================
# Auto-labeling Endpoints
# ============================================================================

@router.post(
    "/faces/auto-label",
    response_model=AutoLabelResponse,
    summary="Auto-label faces",
    description="Automatically label faces based on existing persons",
    status_code=status.HTTP_202_ACCEPTED
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="auto_label", max_calls=10, period=60)
async def auto_label_faces(
    request: AutoLabelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline)
):
    """
    Auto-label faces using existing persons.
    
    Process:
    1. Get unlabeled faces from album
    2. For each person, find similar faces
    3. Auto-label if similarity above threshold
    
    Returns count of labeled faces.
    """
    # Verify album access
    album = db.query(Album).filter(Album.id == request.album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {request.album_id} not found"
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
    
    # Get persons to use as reference
    person_query = db.query(Person).filter(Person.album_id == request.album_id)
    if request.person_ids:
        person_query = person_query.filter(Person.id.in_(request.person_ids))
    
    persons = person_query.all()
    
    if not persons:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No persons found for auto-labeling"
        )
    
    # Get unlabeled faces
    faces_query = db.query(Face).join(Photo).filter(
        Photo.album_id == request.album_id,
        Face.embedding.isnot(None)
    )
    
    if request.unlabeled_only:
        faces_query = faces_query.outerjoin(FacePerson).filter(FacePerson.face_id.is_(None))
    
    faces = faces_query.limit(request.max_faces).all()
    
    if not faces:
        return AutoLabelResponse(
            message="No unlabeled faces found",
            labeled_count=0,
            skipped_count=0,
            face_ids=[],
            person_assignments={}
        )
    
    # Auto-label
    labeled_count = 0
    skipped_count = 0
    labeled_faces = []
    assignments = {}
    
    for face in faces:
        face_emb = np.array(face.embedding, dtype=np.float32)
        best_match = None
        best_score = 0.0
        
        # Find best matching person
        for person in persons:
            # Get person's face embeddings
            person_faces = db.query(Face).join(FacePerson).filter(
                FacePerson.person_id == person.id,
                Face.embedding.isnot(None)
            ).all()
            
            if not person_faces:
                continue
            
            # Calculate average similarity
            person_embs = np.array([f.embedding for f in person_faces], dtype=np.float32)
            similarities = np.dot(person_embs, face_emb) / (
                np.linalg.norm(person_embs, axis=1) * np.linalg.norm(face_emb)
            )
            avg_similarity = float(np.mean(similarities))
            
            if avg_similarity > best_score:
                best_score = avg_similarity
                best_match = person
        
        # Label if above threshold
        if best_match and best_score >= request.min_confidence:
            mapping = FacePerson(
                face_id=face.id,
                person_id=best_match.id,
                is_manual=False
            )
            db.add(mapping)
            labeled_count += 1
            labeled_faces.append(face.id)
            assignments[str(face.id)] = best_match.id
        else:
            skipped_count += 1
    
    db.commit()
    
    logger.info(f"Auto-labeled {labeled_count} faces in album {request.album_id}")
    
    return AutoLabelResponse(
        message=f"Auto-labeled {labeled_count} faces",
        labeled_count=labeled_count,
        skipped_count=skipped_count,
        face_ids=labeled_faces,
        person_assignments=assignments
    )


@router.get(
    "/faces/label-suggestions",
    response_model=LabelSuggestionsResponse,
    summary="Get labeling suggestions",
    description="Get AI-powered suggestions for unlabeled faces"
)
@cache_result(ttl=300)
async def get_label_suggestions(
    album_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    min_confidence: float = Query(0.75, ge=0.6, le=0.95),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Get labeling suggestions for unlabeled faces.
    
    Returns:
    - Face
    - Suggested person
    - Confidence level
    - Reasoning
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
    
    # Get persons
    persons = db.query(Person).filter(Person.album_id == album_id).all()
    
    if not persons:
        return LabelSuggestionsResponse(
            album_id=album_id,
            suggestions=[],
            total=0
        )
    
    # Get unlabeled faces
    faces = db.query(Face).join(Photo).outerjoin(FacePerson).filter(
        Photo.album_id == album_id,
        Face.embedding.isnot(None),
        FacePerson.face_id.is_(None)
    ).limit(limit).all()
    
    suggestions = []
    
    for face in faces:
        face_emb = np.array(face.embedding, dtype=np.float32)
        best_match = None
        best_score = 0.0
        
        # Find best matching person
        for person in persons:
            person_faces = db.query(Face).join(FacePerson).filter(
                FacePerson.person_id == person.id,
                Face.embedding.isnot(None)
            ).all()
            
            if not person_faces:
                continue
            
            person_embs = np.array([f.embedding for f in person_faces], dtype=np.float32)
            similarities = np.dot(person_embs, face_emb) / (
                np.linalg.norm(person_embs, axis=1) * np.linalg.norm(face_emb)
            )
            avg_similarity = float(np.mean(similarities))
            
            if avg_similarity > best_score:
                best_score = avg_similarity
                best_match = person
        
        # Add suggestion if above threshold
        if best_match and best_score >= min_confidence:
            thumbnail_url = None
            if face.thumbnail_s3_key:
                thumbnail_url = s3_service.generate_presigned_download_url(
                    face.thumbnail_s3_key, expires_in=3600
                )
            
            # Determine confidence level
            if best_score >= 0.9:
                confidence = "high"
                reasoning = f"Very high similarity ({best_score:.2%}) to known faces"
            elif best_score >= 0.8:
                confidence = "medium"
                reasoning = f"Good similarity ({best_score:.2%}) to known faces"
            else:
                confidence = "low"
                reasoning = f"Moderate similarity ({best_score:.2%}), review recommended"
            
            suggestions.append(LabelSuggestion(
                face_id=face.id,
                photo_id=face.photo_id,
                thumbnail_url=thumbnail_url,
                person_id=best_match.id,
                person_name=best_match.name,
                similarity_score=best_score,
                confidence=confidence,
                reasoning=reasoning
            ))
    
    return LabelSuggestionsResponse(
        album_id=album_id,
        suggestions=suggestions,
        total=len(suggestions)
    )


@router.post(
    "/faces/{face_id}/confirm-label",
    response_model=FaceResponse,
    summary="Confirm auto-label",
    description="Accept and apply suggested label"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="confirm_label", max_calls=100, period=60)
async def confirm_face_label(
    face_id: UUID,
    request: ConfirmLabelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """Confirm and apply suggested label."""
    face = await get_face_or_404(face_id, db, current_user)
    
    # Check if already labeled
    existing = db.query(FacePerson).filter(FacePerson.face_id == face_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Face already labeled"
        )
    
    # Verify person exists
    person = db.query(Person).filter(Person.id == request.person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found"
        )
    
    # Create mapping
    mapping = FacePerson(
        face_id=face_id,
        person_id=request.person_id,
        is_manual=True  # User confirmed
    )
    db.add(mapping)
    db.commit()
    
    logger.info(f"Confirmed label for face {face_id} -> person {request.person_id}")
    
    return serialize_face(face, s3_service)


@router.post(
    "/faces/{face_id}/reject-label",
    response_model=Dict[str, Any],
    summary="Reject auto-label",
    description="Reject suggested label"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="reject_label", max_calls=100, period=60)
async def reject_face_label(
    face_id: UUID,
    reason: Optional[str] = Body(None, description="Rejection reason"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Reject suggested label.
    
    This helps improve future suggestions by learning from rejections.
    """
    face = await get_face_or_404(face_id, db, current_user)
    
    # TODO: Store rejection for ML feedback
    logger.info(f"Rejected label suggestion for face {face_id}: {reason}")
    
    return {
        "message": "Label suggestion rejected",
        "face_id": str(face_id)
    }


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