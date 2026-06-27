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
    sharing_code: Optional[str] = Query(None, description="Optional sharing code for album access"),
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
    
    # Verify authorization if album_id is specified
    if album_id:
        album = db.query(Album).filter(Album.id == album_id).first()
        if not album:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Album not found"
            )
        is_authorized = (
            current_user.role == 'admin' or
            album.photographer_id == current_user.id or
            album.is_public or
            (sharing_code and album.sharing_code == sharing_code)
        )
        if not is_authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this album"
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
        
        # Query and filter strictly by album_id if provided
        faces_query = db.query(Face).filter(Face.id.in_(face_ids))
        if album_id:
            faces_query = faces_query.join(Photo).filter(Photo.album_id == album_id)
        faces = faces_query.all()
        
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


@router.post(
    "/detect-live",
    summary="Detect faces and landmarks live",
    description="Analyze a front-camera selfie in real-time, returning bounding box and landmarks using RetinaFace"
)
async def detect_live(
    file: UploadFile = File(...),
    pipeline: FacePipeline = Depends(get_pipeline)
):
    """
    Directly run the RetinaFace detector on the uploaded image.
    Returns whether a face was detected, the confidence, the bounding box, and the 5 facial landmarks.
    """
    if file.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are supported"
        )
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )
    
    # Convert BGR to RGB for RetinaFace
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run the detector
    detected_faces = pipeline.detector.detect(image_rgb)
    
    if not detected_faces:
        return {"detected": False}
    
    # Return the highest-confidence face
    best_face = detected_faces[0]
    
    landmarks_list = []
    if best_face.landmarks is not None:
        # Convert numpy array to list of floats
        landmarks_list = best_face.landmarks.tolist()
        
    return {
        "detected": True,
        "bbox": list(best_face.bbox),
        "confidence": float(best_face.confidence),
        "landmarks": landmarks_list
    }


# ============================================================================
# Face Reprocessing Endpoints
# ============================================================================



