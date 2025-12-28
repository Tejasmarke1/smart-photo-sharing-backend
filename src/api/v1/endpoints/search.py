"""
Advanced Face Search API Endpoints
====================================

Production-grade search endpoints for face recognition.

Features:
- Multi-face search (family/group search)
- Cross-album search
- Contextual search with metadata
- Progressive search (fast to refined)
- Threshold optimization
- Search analytics and feedback
- ML model improvement from feedback
"""

from typing import List, Optional, Dict, Any
import json
import base64
from uuid import UUID, uuid4
import logging
from datetime import datetime
import time

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
    Body
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
import numpy as np
import cv2

from src.db.base import get_db
from src.models.face import Face
from src.models.face_person import FacePerson
from src.models.person import Person
from src.models.photo import Photo
from src.models.album import Album
from src.models.user import User
from src.models.search_history import SearchHistory
from src.api.deps import get_current_user, require_roles
from src.services.face.pipeline import FacePipeline, create_pipeline
from src.services.storage.s3 import S3Service
from src.core.cache import cache_result
from src.core.rate_limiter import rate_limit
from src.schemas.search import (
    MultiFaceSearchRequest,
    PersonSearchRequest,
    CrossAlbumSearchRequest,
    ThresholdScanRequest,
    ThresholdScanResponse,
    ThresholdScanResult,
    ContextualSearchRequest,
    ProgressiveSearchRequest,
    ProgressiveSearchResponse,
    ProgressiveSearchStage,
    SearchResponse,
    SearchResult,
    PersonSearchResult,
    CrossAlbumSearchResponse,
    SearchHistoryResponse,
    SearchHistoryEntry,
    SearchFeedback,
    SearchFeedbackResponse,
    SearchImproveRequest,
    SearchImproveResponse
)

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
        logger.info("Face pipeline initialized for search")
    return _pipeline


# ============================================================================
# Helper Functions
# ============================================================================

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image data: {str(e)}"
        )


def get_face_embedding(
    face_id: Optional[UUID],
    selfie_image: Optional[str],
    db: Session,
    pipeline: FacePipeline
) -> np.ndarray:
    """Get face embedding from face_id or selfie image."""
    if face_id:
        face = db.query(Face).filter(Face.id == face_id).first()
        if not face or face.embedding is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face not found or has no embedding"
            )
        return np.array(face.embedding, dtype=np.float32)
    
    elif selfie_image:
        image = decode_base64_image(selfie_image)
        # Detect face and extract embedding
        faces = pipeline.detect_faces(image)
        if not faces:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in selfie"
            )
        return faces[0]['embedding']
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either face_id or selfie_image must be provided"
        )


def record_search(
    db: Session,
    user_id: UUID,
    search_type: str,
    query_params: Dict[str, Any],
    result_count: int,
    processing_time_ms: int,
    album_ids: Optional[List[UUID]] = None,
    query_face_ids: Optional[List[UUID]] = None,
    avg_similarity: Optional[float] = None
) -> SearchHistory:
    """Record search in history for analytics."""
    search_record = SearchHistory(
        user_id=user_id,
        search_type=search_type,
        query_params=query_params,
        result_count=result_count,
        processing_time_ms=processing_time_ms,
        album_ids=[str(aid) for aid in album_ids] if album_ids else None,
        query_face_ids=[str(fid) for fid in query_face_ids] if query_face_ids else None,
        avg_similarity=avg_similarity
    )
    db.add(search_record)
    db.commit()
    db.refresh(search_record)
    return search_record


def build_search_result(
    face: Face,
    similarity_score: float,
    rank: int,
    s3_service: S3Service,
    db: Session
) -> SearchResult:
    """Build SearchResult from Face object."""
    photo = db.query(Photo).filter(Photo.id == face.photo_id).first()
    album = db.query(Album).filter(Album.id == photo.album_id).first() if photo else None
    
    thumbnail_url = None
    if face.thumbnail_s3_key:
        thumbnail_url = s3_service.generate_presigned_download_url(
            face.thumbnail_s3_key, expires_in=3600
        )
    
    person_id = None
    person_name = None
    if face.person_mapping:
        person_id = face.person_mapping.person_id
        person_name = face.person_mapping.person.name if face.person_mapping.person else None
    
    # Calculate face quality
    face_quality = None
    if face.blur_score is not None and face.brightness_score is not None:
        face_quality = (face.blur_score * 0.5 + face.brightness_score * 0.5)
    
    return SearchResult(
        photo_id=face.photo_id,
        face_id=face.id,
        similarity_score=similarity_score,
        thumbnail_url=thumbnail_url,
        person_id=person_id,
        person_name=person_name,
        photo_date=photo.created_at if photo else None,
        album_id=album.id if album else None,
        album_name=album.title if album else None,
        face_confidence=face.confidence,
        face_quality=face_quality,
        rank=rank
    )


# ============================================================================
# Multi-face Search Endpoints
# ============================================================================

@router.post(
    "/multi-face",
    response_model=SearchResponse,
    summary="Multi-face search",
    description="Search with multiple faces (family/group search)"
)
@rate_limit(key_prefix="multi_face_search", max_calls=20, period=60)
async def search_multi_face(
    request: MultiFaceSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Search using multiple faces (e.g., family members).
    
    Aggregation modes:
    - union: Return all matches from any face
    - intersection: Return only photos containing all faces
    - weighted: Weight results by how many faces match
    
    Use case: Find photos containing your family/group.
    """
    start_time = time.time()
    
    # Get embeddings for all query faces
    embeddings = []
    query_face_ids = []
    
    if request.face_ids:
        for face_id in request.face_ids:
            face = db.query(Face).filter(Face.id == face_id).first()
            if face and face.embedding:
                embeddings.append(np.array(face.embedding, dtype=np.float32))
                query_face_ids.append(face_id)
    
    if request.selfie_images:
        for img_str in request.selfie_images:
            image = decode_base64_image(img_str)
            faces = pipeline.detect_faces(image)
            if faces:
                embeddings.append(faces[0]['embedding'])
    
    if not embeddings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid faces provided"
        )
    
    # Search with each embedding
    all_results = {}  # face_id -> {query_idx: similarity}
    
    for idx, embedding in enumerate(embeddings):
        search_results = pipeline.search_engine.search(
            embedding,
            k=request.k,
            threshold=request.threshold
        )
        
        for result in search_results:
            face_id = UUID(result['face_id'])
            if face_id not in all_results:
                all_results[face_id] = {}
            all_results[face_id][idx] = result['score']
    
    # Apply aggregation
    final_results = []
    
    if request.aggregation == 'union':
        # All matches from any face
        for face_id, scores in all_results.items():
            max_score = max(scores.values())
            final_results.append((face_id, max_score))
    
    elif request.aggregation == 'intersection':
        # Only faces matching ALL query faces
        for face_id, scores in all_results.items():
            if len(scores) == len(embeddings):
                avg_score = sum(scores.values()) / len(scores)
                final_results.append((face_id, avg_score))
    
    elif request.aggregation == 'weighted':
        # Weight by number of matches
        for face_id, scores in all_results.items():
            match_count = len(scores)
            avg_score = sum(scores.values()) / len(scores)
            weighted_score = avg_score * (match_count / len(embeddings))
            final_results.append((face_id, weighted_score))
    
    # Sort by score
    final_results.sort(key=lambda x: x[1], reverse=True)
    
    # Fetch faces and build response
    face_ids = [fid for fid, _ in final_results[:request.k]]
    faces = db.query(Face).filter(Face.id.in_(face_ids)).all()
    face_lookup = {f.id: f for f in faces}
    
    # Filter by albums if specified
    if request.album_ids:
        photos_in_albums = db.query(Photo.id).filter(
            Photo.album_id.in_(request.album_ids)
        ).all()
        valid_photo_ids = {p[0] for p in photos_in_albums}
        final_results = [
            (fid, score) for fid, score in final_results
            if face_lookup.get(fid) and face_lookup[fid].photo_id in valid_photo_ids
        ]
    
    # Build results
    results = []
    for rank, (face_id, score) in enumerate(final_results[:request.k], 1):
        face = face_lookup.get(face_id)
        if face:
            results.append(build_search_result(face, score, rank, s3_service, db))
    
    processing_time = int((time.time() - start_time) * 1000)
    query_id = uuid4()
    
    # Record search
    user_id = getattr(current_user, 'id', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
    
    record_search(
        db=db,
        user_id=user_id,
        search_type='multi_face',
        query_params=request.model_dump(),
        result_count=len(results),
        processing_time_ms=processing_time,
        album_ids=request.album_ids,
        query_face_ids=query_face_ids
    )
    
    return SearchResponse(
        query_id=query_id,
        total_results=len(results),
        results=results,
        processing_time_ms=processing_time,
        search_params=request.model_dump()
    )


@router.post(
    "/by-person",
    response_model=List[PersonSearchResult],
    summary="Search by person",
    description="Find all photos containing a specific person"
)
@rate_limit(key_prefix="person_search", max_calls=50, period=60)
@cache_result(ttl=300)
async def search_by_person(
    request: PersonSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Search all photos containing a person.
    
    Returns photos with:
    - Face count in that photo
    - Photo metadata
    - Quality scores
    
    Useful for: Personal galleries, person-centric views.
    """
    start_time = time.time()
    
    # Verify person exists
    person = db.query(Person).filter(Person.id == request.person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found"
        )
    
    # Get all faces for this person
    query = db.query(Face).join(FacePerson).filter(
        FacePerson.person_id == request.person_id
    )
    
    # Apply quality filter
    if request.min_face_quality:
        # Filter by average of blur and brightness
        query = query.filter(
            ((Face.blur_score + Face.brightness_score) / 2) >= request.min_face_quality
        )
    
    faces = query.all()
    
    # Group by photo
    photo_faces = {}
    for face in faces:
        if face.photo_id not in photo_faces:
            photo_faces[face.photo_id] = []
        photo_faces[face.photo_id].append(face)
    
    # Get photos
    photo_ids = list(photo_faces.keys())
    photos_query = db.query(Photo).filter(Photo.id.in_(photo_ids))
    
    # Filter by albums if specified
    if request.album_ids:
        photos_query = photos_query.filter(Photo.album_id.in_(request.album_ids))
    
    # Filter by date range
    if request.date_from:
        try:
            date_from = datetime.fromisoformat(request.date_from.replace('Z', '+00:00'))
            photos_query = photos_query.filter(Photo.created_at >= date_from)
        except ValueError:
            pass
    
    if request.date_to:
        try:
            date_to = datetime.fromisoformat(request.date_to.replace('Z', '+00:00'))
            photos_query = photos_query.filter(Photo.created_at <= date_to)
        except ValueError:
            pass
    
    photos = photos_query.limit(request.limit).all()
    
    # Build results
    results = []
    for photo in photos:
        album = db.query(Album).filter(Album.id == photo.album_id).first()
        faces_in_photo = photo_faces.get(photo.id, [])
        
        # Calculate average quality
        qualities = []
        for face in faces_in_photo:
            if face.blur_score is not None and face.brightness_score is not None:
                qualities.append((face.blur_score + face.brightness_score) / 2)
        
        avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
        
        # Get thumbnail
        thumbnail_url = None
        if photo.s3_key:
            thumbnail_url = s3_service.generate_presigned_download_url(
                photo.s3_key, expires_in=3600
            )
        
        results.append(PersonSearchResult(
            photo_id=photo.id,
            album_id=photo.album_id,
            album_name=album.title if album else "Unknown",
            face_ids=[f.id for f in faces_in_photo],
            face_count=len(faces_in_photo),
            thumbnail_url=thumbnail_url,
            photo_date=photo.created_at,
            avg_quality=avg_quality
        ))
    
    processing_time = int((time.time() - start_time) * 1000)
    
    # Record search
    user_id = getattr(current_user, 'id', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
    
    record_search(
        db=db,
        user_id=user_id,
        search_type='by_person',
        query_params=request.model_dump(),
        result_count=len(results),
        processing_time_ms=processing_time,
        album_ids=request.album_ids
    )
    
    logger.info(f"Person search for {request.person_id}: {len(results)} photos")
    return results


@router.post(
    "/cross-album",
    response_model=CrossAlbumSearchResponse,
    summary="Cross-album search",
    description="Search for a face across multiple albums"
)
@rate_limit(key_prefix="cross_album_search", max_calls=30, period=60)
async def search_cross_album(
    request: CrossAlbumSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Search across multiple albums.
    
    Useful for:
    - Finding someone across events
    - Cross-album face recognition
    - Multi-album galleries
    
    Results can be grouped by album or unified.
    """
    start_time = time.time()
    
    # Get query embedding
    embedding = get_face_embedding(
        request.face_id,
        request.selfie_image,
        db,
        pipeline
    )
    
    # Search
    search_results = pipeline.search_engine.search(
        embedding,
        k=request.limit_per_album * len(request.album_ids),
        threshold=request.threshold
    )
    
    # Get faces
    face_ids = [UUID(r['face_id']) for r in search_results]
    faces = db.query(Face).filter(Face.id.in_(face_ids)).all()
    face_lookup = {str(f.id): f for f in faces}
    
    # Filter by albums
    photos_by_album = {}
    for album_id in request.album_ids:
        photos = db.query(Photo.id).filter(Photo.album_id == album_id).all()
        photos_by_album[str(album_id)] = {p[0] for p in photos}
    
    # Group results by album
    results_by_album = {str(aid): [] for aid in request.album_ids}
    all_results = []
    rank = 1
    
    for result in search_results:
        face = face_lookup.get(result['face_id'])
        if not face:
            continue
        
        # Find which album this face belongs to
        for album_id_str, photo_ids in photos_by_album.items():
            if face.photo_id in photo_ids:
                if len(results_by_album[album_id_str]) < request.limit_per_album:
                    search_result = build_search_result(
                        face, result['score'], rank, s3_service, db
                    )
                    results_by_album[album_id_str].append(search_result)
                    all_results.append(search_result)
                    rank += 1
                break
    
    # Get top results across all albums
    all_results.sort(key=lambda x: x.similarity_score, reverse=True)
    top_results = all_results[:50]  # Top 50 overall
    
    processing_time = int((time.time() - start_time) * 1000)
    query_id = uuid4()
    
    # Record search
    user_id = getattr(current_user, 'id', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
    
    record_search(
        db=db,
        user_id=user_id,
        search_type='cross_album',
        query_params=request.model_dump(),
        result_count=len(all_results),
        processing_time_ms=processing_time,
        album_ids=request.album_ids,
        query_face_ids=[request.face_id] if request.face_id else None
    )
    
    return CrossAlbumSearchResponse(
        query_id=query_id,
        total_results=len(all_results),
        albums_searched=len(request.album_ids),
        results_by_album=results_by_album,
        top_results=top_results,
        processing_time_ms=processing_time
    )


# ============================================================================
# Similarity Search Variations
# ============================================================================

@router.post(
    "/threshold-scan",
    response_model=ThresholdScanResponse,
    summary="Find optimal threshold",
    description="Scan different thresholds to find optimal similarity threshold"
)
@rate_limit(key_prefix="threshold_scan", max_calls=10, period=60)
async def scan_threshold(
    request: ThresholdScanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Find optimal similarity threshold.
    
    Process:
    1. Search with multiple thresholds
    2. Count results at each threshold
    3. Recommend threshold closest to target count
    
    Use case: Calibrate search sensitivity.
    """
    # Get query embedding
    embedding = get_face_embedding(
        request.face_id,
        request.selfie_image,
        db,
        pipeline
    )
    
    # Scan thresholds
    scan_results = []
    thresholds = np.arange(
        request.min_threshold,
        request.max_threshold + request.step,
        request.step
    )
    
    for threshold in thresholds:
        results = pipeline.search_engine.search(
            embedding,
            k=500,  # Get many results
            threshold=float(threshold)
        )
        
        # Filter by album
        face_ids = [UUID(r['face_id']) for r in results]
        faces_in_album = db.query(Face.id).join(Photo).filter(
            Photo.album_id == request.album_id,
            Face.id.in_(face_ids)
        ).all()
        valid_ids = {f[0] for f in faces_in_album}
        
        filtered_results = [r for r in results if UUID(r['face_id']) in valid_ids]
        
        if filtered_results:
            avg_sim = sum(r['score'] for r in filtered_results) / len(filtered_results)
        else:
            avg_sim = 0.0
        
        # Get sample results
        sample = []
        for r in filtered_results[:5]:
            face = db.query(Face).filter(Face.id == UUID(r['face_id'])).first()
            if face:
                thumb_url = None
                if face.thumbnail_s3_key:
                    thumb_url = s3_service.generate_presigned_download_url(
                        face.thumbnail_s3_key, expires_in=3600
                    )
                sample.append({
                    'face_id': str(face.id),
                    'similarity': r['score'],
                    'thumbnail_url': thumb_url
                })
        
        scan_results.append(ThresholdScanResult(
            threshold=float(threshold),
            result_count=len(filtered_results),
            avg_similarity=avg_sim,
            sample_results=sample
        ))
    
    # Find recommended threshold
    best_threshold = request.min_threshold
    min_diff = float('inf')
    
    for result in scan_results:
        diff = abs(result.result_count - request.target_result_count)
        if diff < min_diff:
            min_diff = diff
            best_threshold = result.threshold
    
    # Generate reasoning
    reasoning = f"Threshold {best_threshold:.2f} yields {[r for r in scan_results if r.threshold == best_threshold][0].result_count} results, "
    reasoning += f"closest to target of {request.target_result_count}. "
    reasoning += f"Lower threshold = more results (higher recall), higher threshold = fewer results (higher precision)."
    
    logger.info(f"Threshold scan: recommended {best_threshold} for target {request.target_result_count}")
    
    return ThresholdScanResponse(
        recommended_threshold=best_threshold,
        scan_results=scan_results,
        reasoning=reasoning
    )


@router.post(
    "/with-context",
    response_model=SearchResponse,
    summary="Contextual search",
    description="Search with metadata context (date, location, tags)"
)
@rate_limit(key_prefix="contextual_search", max_calls=30, period=60)
async def search_with_context(
    request: ContextualSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Search with metadata context.
    
    Combines:
    - Face similarity (similarity_weight)
    - Context matching (context_weight)
    
    Context includes:
    - Date range
    - Location proximity
    - Event tags
    - Photo quality
    
    Use case: "Find me at wedding in summer 2024"
    """
    start_time = time.time()
    
    # Get query embedding
    embedding = get_face_embedding(
        request.face_id,
        request.selfie_image,
        db,
        pipeline
    )
    
    # Face similarity search
    search_results = pipeline.search_engine.search(
        embedding,
        k=500,  # Get more for filtering
        threshold=request.threshold
    )
    
    # Filter by album
    face_ids = [UUID(r['face_id']) for r in search_results]
    faces_query = db.query(Face).join(Photo).filter(
        Photo.album_id == request.album_id,
        Face.id.in_(face_ids)
    )
    
    # Apply context filters
    if request.date_range:
        if 'from' in request.date_range:
            try:
                date_from = datetime.fromisoformat(request.date_range['from'].replace('Z', '+00:00'))
                faces_query = faces_query.filter(Photo.created_at >= date_from)
            except ValueError:
                pass
        if 'to' in request.date_range:
            try:
                date_to = datetime.fromisoformat(request.date_range['to'].replace('Z', '+00:00'))
                faces_query = faces_query.filter(Photo.created_at <= date_to)
            except ValueError:
                pass
    
    if request.photo_quality_min:
        # TODO: Add photo quality score to Photo model
        pass
    
    faces = faces_query.all()
    face_lookup = {str(f.id): f for f in faces}
    
    # Calculate combined scores
    scored_results = []
    for result in search_results:
        face = face_lookup.get(result['face_id'])
        if not face:
            continue
        
        similarity_score = result['score']
        context_score = 1.0  # Base context score
        
        # TODO: Calculate context score based on metadata
        # For now, use simple filters
        
        # Combined score
        final_score = (
            similarity_score * request.similarity_weight +
            context_score * request.context_weight
        )
        
        scored_results.append((face, final_score))
    
    # Sort by combined score
    scored_results.sort(key=lambda x: x[1], reverse=True)
    scored_results = scored_results[:request.limit]
    
    # Build results
    results = []
    for rank, (face, score) in enumerate(scored_results, 1):
        result = build_search_result(face, score, rank, s3_service, db)
        result.relevance_score = score
        results.append(result)
    
    processing_time = int((time.time() - start_time) * 1000)
    query_id = uuid4()
    
    # Record search
    user_id = getattr(current_user, 'id', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
    
    record_search(
        db=db,
        user_id=user_id,
        search_type='contextual',
        query_params=request.model_dump(),
        result_count=len(results),
        processing_time_ms=processing_time,
        album_ids=[request.album_id]
    )
    
    return SearchResponse(
        query_id=query_id,
        total_results=len(results),
        results=results,
        processing_time_ms=processing_time,
        search_params=request.model_dump()
    )


@router.post(
    "/progressive",
    response_model=ProgressiveSearchResponse,
    summary="Progressive search",
    description="Progressive search (fast initial, then refine)"
)
@rate_limit(key_prefix="progressive_search", max_calls=20, period=60)
async def search_progressive(
    request: ProgressiveSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Progressive search: start fast, refine iteratively.
    
    Stages:
    1. High threshold (fast, high precision)
    2. Medium threshold (balanced)
    3. Low threshold (comprehensive, high recall)
    
    Stops early if enough results found or timeout reached.
    
    Use case: Interactive search with progressive loading.
    """
    start_time = time.time()
    
    # Get query embedding
    embedding = get_face_embedding(
        request.face_id,
        request.selfie_image,
        db,
        pipeline
    )
    
    # Calculate thresholds for each stage
    thresholds = np.linspace(
        request.initial_threshold,
        request.final_threshold,
        request.stages
    )
    
    stages = []
    all_results = {}  # face_id -> (face, score)
    completed_early = False
    early_reason = None
    
    for stage_num, threshold in enumerate(thresholds, 1):
        stage_start = time.time()
        
        # Check timeout
        elapsed = (stage_start - start_time) * 1000
        if elapsed > request.timeout_seconds * 1000:
            completed_early = True
            early_reason = "Timeout reached"
            break
        
        # Search at this threshold
        results = pipeline.search_engine.search(
            embedding,
            k=request.max_results,
            threshold=float(threshold)
        )
        
        # Filter by album
        new_count = 0
        for result in results:
            face_id = result['face_id']
            if face_id not in all_results:
                face = db.query(Face).join(Photo).filter(
                    Face.id == UUID(face_id),
                    Photo.album_id == request.album_id
                ).first()
                
                if face:
                    all_results[face_id] = (face, result['score'])
                    new_count += 1
        
        stage_time = int((time.time() - stage_start) * 1000)
        
        stages.append(ProgressiveSearchStage(
            stage=stage_num,
            threshold=float(threshold),
            results_count=len(all_results),
            processing_time_ms=stage_time,
            is_complete=True
        ))
        
        # Check if we have enough results
        if len(all_results) >= request.max_results:
            completed_early = True
            early_reason = f"Target of {request.max_results} results reached"
            break
    
    # Build final results
    sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
    results = []
    
    for rank, (face, score) in enumerate(sorted_results[:request.max_results], 1):
        results.append(build_search_result(face, score, rank, s3_service, db))
    
    total_time = int((time.time() - start_time) * 1000)
    
    # Record search
    user_id = getattr(current_user, 'id', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
    
    record_search(
        db=db,
        user_id=user_id,
        search_type='progressive',
        query_params=request.model_dump(),
        result_count=len(results),
        processing_time_ms=total_time,
        album_ids=[request.album_id]
    )
    
    logger.info(f"Progressive search: {len(stages)} stages, {len(results)} results in {total_time}ms")
    
    return ProgressiveSearchResponse(
        stages=stages,
        total_results=len(results),
        total_time_ms=total_time,
        results=[r.model_dump() for r in results],
        completed_early=completed_early,
        reason=early_reason
    )


# ============================================================================
# Search Analytics Endpoints
# ============================================================================

@router.get(
    "/history",
    response_model=SearchHistoryResponse,
    summary="Search history",
    description="Get user's search history"
)
@rate_limit(key_prefix="search_history", max_calls=50, period=60)
async def get_search_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    search_type: Optional[str] = Query(None, description="Filter by search type"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user's search history.
    
    Includes:
    - Search parameters
    - Result counts
    - Timestamps
    - Feedback status
    """
    user_id = getattr(current_user, 'id', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
    
    query = db.query(SearchHistory).filter(SearchHistory.user_id == user_id)
    
    if search_type:
        query = query.filter(SearchHistory.search_type == search_type)
    
    total = query.count()
    history = query.order_by(desc(SearchHistory.created_at)).offset(skip).limit(limit).all()
    
    entries = []
    for record in history:
        entries.append(SearchHistoryEntry(
            id=record.id,
            search_type=record.search_type,
            query_params=record.query_params,
            result_count=record.result_count,
            created_at=record.created_at,
            feedback_given=record.feedback_given,
            avg_relevance=record.avg_relevance_score
        ))
    
    return SearchHistoryResponse(
        total=total,
        entries=entries,
        skip=skip,
        limit=limit
    )


@router.post(
    "/{search_id}/feedback",
    response_model=SearchFeedbackResponse,
    summary="Submit search feedback",
    description="Provide feedback on search results"
)
@rate_limit(key_prefix="search_feedback", max_calls=50, period=60)
async def submit_search_feedback(
    search_id: UUID,
    feedback: SearchFeedback,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submit feedback on search results.
    
    Feedback includes:
    - Relevant results (correct)
    - Irrelevant results (incorrect)
    - Missing expected results
    - General comments
    
    Used to improve search quality.
    """
    user_id = getattr(current_user, 'id', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
    
    # Find search record
    search = db.query(SearchHistory).filter(
        SearchHistory.id == search_id,
        SearchHistory.user_id == user_id
    ).first()
    
    if not search:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Search not found"
        )
    
    # Update feedback
    search.feedback_given = True
    search.relevant_face_ids = [str(fid) for fid in feedback.relevant_face_ids]
    search.irrelevant_face_ids = [str(fid) for fid in feedback.irrelevant_face_ids]
    search.missing_expected = feedback.missing_expected
    search.too_many_results = feedback.too_many_results
    search.feedback_comments = feedback.comments
    
    # Calculate relevance score
    total_marked = len(feedback.relevant_face_ids) + len(feedback.irrelevant_face_ids)
    if total_marked > 0:
        relevance = len(feedback.relevant_face_ids) / total_marked
        search.avg_relevance_score = relevance
    
    db.commit()
    
    logger.info(f"Feedback recorded for search {search_id}: {len(feedback.relevant_face_ids)} relevant, {len(feedback.irrelevant_face_ids)} irrelevant")
    
    return SearchFeedbackResponse(
        message="Feedback recorded successfully",
        search_id=search_id,
        feedback_recorded=True,
        improvements_applied=False  # TODO: Implement ML feedback loop
    )


@router.post(
    "/improve",
    response_model=SearchImproveResponse,
    summary="Improve search",
    description="Mark search results as correct/incorrect for ML improvement"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="search_improve", max_calls=30, period=60)
async def improve_search_results(
    request: SearchImproveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Mark search results quality for ML improvement.
    
    Corrections:
    - is_correct: Boolean marking
    - should_rank_higher/lower: Ranking adjustments
    
    This data trains the search ranking model.
    """
    user_id = getattr(current_user, 'id', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
    
    # Verify search exists
    search = db.query(SearchHistory).filter(
        SearchHistory.id == request.search_id,
        SearchHistory.user_id == user_id
    ).first()
    
    if not search:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Search not found"
        )
    
    # Store corrections
    corrections_data = {
        'corrections': request.result_corrections,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if search.extra_data:
        if isinstance(search.extra_data, dict):
            search.extra_data['quality_corrections'] = corrections_data
        else:
            search.extra_data = {'quality_corrections': corrections_data}
    else:
        search.extra_data = {'quality_corrections': corrections_data}
    
    search.feedback_given = True
    
    # Calculate stats
    correct_count = sum(1 for c in request.result_corrections if c.get('is_correct', False))
    total_count = len(request.result_corrections)
    
    if total_count > 0:
        search.avg_relevance_score = correct_count / total_count
    
    db.commit()
    
    # TODO: Implement ML model retraining based on corrections
    logger.info(f"Search improvements: {correct_count}/{total_count} marked correct for search {request.search_id}")
    
    return SearchImproveResponse(
        message="Search improvements recorded",
        corrections_applied=len(request.result_corrections),
        model_updated=False,  # TODO: Implement ML update
        new_threshold_suggestion=None  # TODO: Calculate based on corrections
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
    logger.error(f"Unexpected error in search: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )