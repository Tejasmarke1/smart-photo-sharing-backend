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
        
    # Sync index with DB dynamically if count differs
    if _pipeline and _pipeline.search_engine:
        try:
            from src.db.base import SessionLocal
            from src.models.face import Face
            db = SessionLocal()
            try:
                num_indexed = _pipeline.search_engine.index.ntotal
                db_count = db.query(Face).filter(Face.embedding.isnot(None)).count()
                if num_indexed != db_count:
                    logger.info(f"🔄 Syncing FAISS index (indexed: {num_indexed}, DB: {db_count})")
                    _pipeline.rebuild_index_from_db(db)
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to sync FAISS index: {e}")
            
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
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    # Ensure user_id is a UUID for consistent comparison with ORM UUID columns
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
        bbox=json.loads(face.bbox.replace("'", "\"")) if isinstance(face.bbox, str) else face.bbox,
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

