"""
Production-Grade Person Management API
======================================

Complete person management with security, validation, async operations,
caching, audit logging, and comprehensive error handling.

Features:
- Authentication & Authorization
- Input validation with limits
- Rate limiting per endpoint
- Async background jobs for heavy operations
- Redis caching layer
- Audit logging
- Transaction safety
- Webhook support
- Comprehensive error handling
- OpenAPI documentation
"""

from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
import logging
from functools import wraps

from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    Query, 
    Path,
    BackgroundTasks,
    Request,
    status
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.db.base import get_db
from src.models.user import User
from src.models.person import Person
from src.models.album import Album
from src.api.deps import (
    get_current_user, 
    require_roles,
)
from src.core.cache import cache_result, invalidate_cache, redis_client
from src.core.rate_limiter import rate_limit
from src.repositories.person_repo import PersonRepository
from src.repositories.album_repo import AlbumRepository
from src.services.storage.s3 import S3Service, S3ServiceException
from src.core.audit_log import audit_log
from src.services.webhooks import trigger_webhook
from src.tasks.workers.person_worker import (
    batch_merge_task,
    batch_label_task,
    transfer_person_task
)
from src.schemas.person import (
    PersonCreate,
    PersonUpdate,
    PersonInDB,
    PersonWithFaces,
    PersonWithPhotos,
    PersonListResponse,
    PersonThumbnailResponse,
    PersonMergeRequest,
    PersonMergeResponse,
    BatchLabelRequest,
    BatchLabelResponse,
    BatchMergeRequest,
    BatchMergeResponse,
    PersonSplitRequest,
    PersonSplitResponse,
    PersonTransferRequest,
    PersonTransferResponse,
    SimilarPersonsResponse,
    
)
from src.schemas.face import FaceInDB
router = APIRouter()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)


# =============================================================================
# Constants & Configuration
# =============================================================================

MAX_BATCH_SIZE = 100
MAX_MERGE_SIZE = 50
MAX_LIST_LIMIT = 500
DEFAULT_CACHE_TTL = 300  # 5 minutes
HEAVY_OPERATION_THRESHOLD = 10  # When to use background tasks


# =============================================================================
# Custom Decorators
# =============================================================================

def validate_person_access(func):
    """Decorator to verify user has access to person."""
    @wraps(func)
    async def wrapper(
        person_id: UUID,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
        *args,
        **kwargs
    ):
        repo = PersonRepository(db)
        person = repo.get_by_id(person_id)
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )
        
        # Check if user has access to this person's album
        album_repo = AlbumRepository(db)
        album = album_repo.get_by_id(person.album_id)
        
        if not album:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Album not found"
            )
        
        # Verify access
        if album.photographer_id != current_user.id and current_user.role not in ['admin', 'editor']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this person"
            )
        
        # Add person to kwargs for function to use
        kwargs['person'] = person
        kwargs['db'] = db
        kwargs['current_user'] = current_user
        
        return await func(person_id=person_id, *args, **kwargs)
    
    return wrapper


# =============================================================================
# Enhanced Request Models with Validation
# =============================================================================

class EnhancedPersonCreate(PersonCreate):
    """Enhanced person creation with validation."""
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Name must be at least 2 characters")
        if len(v) > 100:
            raise ValueError("Name must be less than 100 characters")
        return v.strip()


class EnhancedBatchLabelRequest(BatchLabelRequest):
    """Enhanced batch labeling with limits."""
    
    face_ids: List[UUID] = Field(..., min_items=1, max_items=MAX_BATCH_SIZE)
    
    @validator('face_ids')
    def validate_face_ids(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("Duplicate face_ids not allowed")
        return v


class EnhancedBatchMergeRequest(BatchMergeRequest):
    """Enhanced batch merge with limits."""
    
    person_ids: List[UUID] = Field(..., min_items=2, max_items=MAX_MERGE_SIZE)
    
    @validator('person_ids')
    def validate_person_ids(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("Duplicate person_ids not allowed")
        return v


# =============================================================================
# Helper Functions
# =============================================================================

async def verify_album_ownership(
    album_id: UUID,
    user: User,
    db: Session
) -> Album:
    """Verify user owns or has access to album."""
    album_repo = AlbumRepository(db)
    album = album_repo.get_by_id(album_id)
    
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {album_id} not found"
        )
    
    if album.photographer_id != user.id and user.role not in ['admin', 'editor']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this album"
        )
    
    return album


def get_cache_key(resource: str, resource_id: UUID, suffix: str = "") -> str:
    """Generate consistent cache key."""
    return f"person:{resource}:{resource_id}{f':{suffix}' if suffix else ''}"


async def invalidate_person_cache(person_id: UUID, album_id: UUID):
    """Invalidate all cache entries for a person."""
    keys = [
        get_cache_key("details", person_id),
        get_cache_key("faces", person_id),
        get_cache_key("photos", person_id),
        get_cache_key("thumbnail", person_id),
        f"album:persons:{album_id}",
        f"person:similar:{person_id}"
    ]
    
    for key in keys:
        try:
            await invalidate_cache(key)
        except Exception as e:
            logger.warning(f"Failed to invalidate cache key {key}: {e}")


# =============================================================================
# Core Person Operations
# =============================================================================

@router.get("", response_model=PersonListResponse)
@limiter.limit("100/minute")
@cache_result(ttl=DEFAULT_CACHE_TTL)
async def list_persons(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=MAX_LIST_LIMIT, description="Number of records to return"),
    album_id: Optional[UUID] = Query(None, description="Filter by album ID"),
    search: Optional[str] = Query(None, max_length=100, description="Search by name"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all persons with pagination and filtering.
    
    **Features:**
    - Pagination support
    - Filter by album
    - Search by name
    - Caching enabled (5 min)
    - Rate limited (100 req/min)
    
    **Returns:**
    - List of persons
    - Total count
    - Pagination info
    """
    try:
        # Verify album access if filtering
        if album_id:
            await verify_album_ownership(album_id, current_user, db)
        
        repo = PersonRepository(db)
        
        # Get persons with filters
        persons, total = repo.get_all(
            skip=skip,
            limit=limit,
            album_id=album_id,
            search=search,
            user_id=current_user.id if current_user.role != 'admin' else None
        )
        
        # Calculate pagination
        has_more = (skip + len(persons)) < total
        page = (skip // limit) + 1 if limit > 0 else 1
        
        logger.info(
            f"Listed {len(persons)} persons",
            extra={
                "user_id": current_user.id,
                "album_id": str(album_id) if album_id else None,
                "total": total
            }
        )
        
        return PersonListResponse(
            persons=[PersonInDB.model_validate(p) for p in persons],
            total=total,
            page=page,
            page_size=limit,
            has_more=has_more
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list persons: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve persons"
        )


@router.post("", response_model=PersonInDB, status_code=status.HTTP_201_CREATED)
@limiter.limit("50/minute")
@require_roles(['photographer', 'admin'])
async def create_person(
    request: Request,
    person_data: EnhancedPersonCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new person in an album.
    
    **Security:**
    - Requires photographer or admin role
    - Verifies album ownership
    - Rate limited (50 req/min)
    
    **Features:**
    - Input validation
    - Duplicate detection
    - Audit logging
    - Webhook notification
    """
    try:
        # Verify album access
        album = await verify_album_ownership(person_data.album_id, current_user, db)
        
        repo = PersonRepository(db)
        
        # Check for duplicate name in album
        existing = repo.find_by_name(person_data.album_id, person_data.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Person with name '{person_data.name}' already exists in this album"
            )
        
        # Create person
        person = repo.create(person_data)
        db.commit()
        db.refresh(person)
        
        # Invalidate album cache
        await invalidate_cache(f"album:persons:{album.id}")
        
        # Audit log
        audit_log.record(
            user_id=current_user.id,
            action="person.create",
            resource_type="person",
            resource_id=person.id,
            details={
                "name": person.name,
                "album_id": str(album.id)
            }
        )
        
        # Webhook notification (background)
        background_tasks.add_task(
            trigger_webhook,
            event="person.created",
            data={
                "person_id": str(person.id),
                "name": person.name,
                "album_id": str(album.id)
            }
        )
        
        logger.info(f"Created person {person.id} in album {album.id}")
        
        return PersonInDB.model_validate(person)
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error creating person: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create person"
        )


@router.get("/{person_id}", response_model=PersonInDB)
@limiter.limit("200/minute")
@cache_result(ttl=DEFAULT_CACHE_TTL)
async def get_person(
    request: Request,
    person_id: UUID = Path(..., description="Person ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get person details by ID.
    
    **Features:**
    - Access control
    - Caching (5 min)
    - Rate limited (200 req/min)
    """
    try:
        repo = PersonRepository(db)
        person = repo.get_by_id(person_id)
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )
        
        # Verify access
        await verify_album_ownership(person.album_id, current_user, db)
        
        return PersonInDB.model_validate(person)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get person {person_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve person"
        )


@router.patch("/{person_id}", response_model=PersonInDB)
@limiter.limit("50/minute")
@require_roles(['photographer', 'admin'])
async def update_person(
    request: Request,
    person_id: UUID = Path(..., description="Person ID"),
    person_data: PersonUpdate = ...,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update person details.
    
    **Security:**
    - Requires photographer or admin role
    - Verifies ownership
    - Rate limited (50 req/min)
    
    **Features:**
    - Partial updates
    - Duplicate name check
    - Cache invalidation
    - Audit logging
    """
    try:
        repo = PersonRepository(db)
        person = repo.get_by_id(person_id)
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )
        
        # Verify access
        await verify_album_ownership(person.album_id, current_user, db)
        
        # Check for duplicate name if updating name
        if person_data.name and person_data.name != person.name:
            existing = repo.find_by_name(person.album_id, person_data.name)
            if existing and existing.id != person_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Person with name '{person_data.name}' already exists"
                )
        
        # Store old values for audit
        old_values = {
            "name": person.name,
            "phone": person.phone,
            "email": person.email
        }
        
        # Update
        updated_person = repo.update(person_id, person_data)
        db.commit()
        db.refresh(updated_person)
        
        # Invalidate cache
        await invalidate_person_cache(person_id, person.album_id)
        
        # Audit log
        audit_log.record(
            user_id=current_user.id,
            action="person.update",
            resource_type="person",
            resource_id=person_id,
            details={
                "old_values": old_values,
                "new_values": person_data.dict(exclude_unset=True)
            }
        )
        
        logger.info(f"Updated person {person_id}")
        
        return PersonInDB.model_validate(updated_person)
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error updating person: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update person"
        )


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("30/minute")
@require_roles(['photographer', 'admin'])
async def delete_person(
    request: Request,
    person_id: UUID = Path(..., description="Person ID"),
    force: bool = Query(False, description="Force delete (remove all face mappings)"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a person (soft delete by default).
    
    **Security:**
    - Requires photographer or admin role
    - Verifies ownership
    - Rate limited (30 req/min)
    
    **Features:**
    - Soft delete (default)
    - Force delete option
    - Cascade handling
    - Audit logging
    """
    try:
        repo = PersonRepository(db)
        person = repo.get_by_id(person_id)
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )
        
        # Verify access
        await verify_album_ownership(person.album_id, current_user, db)
        
        # Check for face mappings
        face_count = repo.get_face_count(person_id)
        
        if face_count > 0 and not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Person has {face_count} labeled faces. Use force=true to delete anyway."
            )
        
        # Store for audit
        person_data = {
            "name": person.name,
            "album_id": str(person.album_id),
            "face_count": face_count
        }
        
        # Delete
        success = repo.delete(person_id, force=force)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete person"
            )
        
        db.commit()
        
        # Invalidate cache
        await invalidate_person_cache(person_id, person.album_id)
        
        # Audit log
        audit_log.record(
            user_id=current_user.id,
            action="person.delete",
            resource_type="person",
            resource_id=person_id,
            details={
                **person_data,
                "force": force
            }
        )
        
        # Webhook notification (background)
        if background_tasks:
            background_tasks.add_task(
                trigger_webhook,
                event="person.deleted",
                data={
                    "person_id": str(person_id),
                    "album_id": str(person.album_id)
                }
            )
        
        logger.info(f"Deleted person {person_id} (force={force})")
        
        return None
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error deleting person: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete person"
        )


# =============================================================================
# Person Photos and Faces
# =============================================================================

@router.get("/{person_id}/faces", response_model=List[FaceInDB])
@limiter.limit("100/minute")
@cache_result(ttl=DEFAULT_CACHE_TTL)
async def get_person_faces(
    request: Request,
    person_id: UUID = Path(..., description="Person ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    min_quality: Optional[float] = Query(None, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all faces assigned to a person (paginated).
    
    **Features:**
    - Pagination support
    - Quality filtering
    - Caching (5 min)
    - Rate limited (100 req/min)
    """
    try:
        repo = PersonRepository(db)
        person = repo.get_by_id(person_id)
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )
        
        # Verify access
        await verify_album_ownership(person.album_id, current_user, db)
        
        # Get faces with filters
        faces = repo.get_faces(
            person_id,
            skip=skip,
            limit=limit,
            min_quality=min_quality
        )
        
        return [FaceInDB.model_validate(f) for f in faces]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get faces for person {person_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve faces"
        )


@router.get("/{person_id}/photos", response_model=PersonWithPhotos)
@limiter.limit("100/minute")
@cache_result(ttl=DEFAULT_CACHE_TTL)
async def get_person_photos(
    request: Request,
    person_id: UUID = Path(..., description="Person ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all photos containing this person (paginated).
    
    **Returns:**
    - Person details
    - Photo IDs (paginated)
    - Total photo count
    """
    try:
        repo = PersonRepository(db)
        person = repo.get_by_id(person_id)
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )
        
        # Verify access
        await verify_album_ownership(person.album_id, current_user, db)
        
        # Get photos with pagination
        photo_ids, total = repo.get_photos(person_id, skip=skip, limit=limit)
        
        person_dict = PersonInDB.model_validate(person).model_dump()
        person_dict['photo_ids'] = photo_ids
        person_dict['photo_count'] = total
        
        return PersonWithPhotos(**person_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get photos for person {person_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve photos"
        )


@router.get("/{person_id}/thumbnail", response_model=PersonThumbnailResponse)
@limiter.limit("200/minute")
@cache_result(ttl=DEFAULT_CACHE_TTL)
async def get_person_thumbnail(
    request: Request,
    person_id: UUID = Path(..., description="Person ID"),
    generate_url: bool = Query(True, description="Generate presigned S3 URL"),
    size: str = Query("medium", regex="^(small|medium|large)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get best representative face thumbnail for person.
    
    **Features:**
    - Returns highest quality face
    - Multiple size options
    - Fallback to default avatar
    - Presigned S3 URLs
    - Caching (5 min)
    """
    try:
        repo = PersonRepository(db)
        person = repo.get_by_id(person_id)
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )
        
        # Verify access
        await verify_album_ownership(person.album_id, current_user, db)
        
        # Get best face
        face = repo.get_thumbnail(person_id)
        
        if not face or not face.thumbnail_s3_key:
            # Return default avatar
            return PersonThumbnailResponse(
                person_id=person_id,
                thumbnail_url="/static/avatars/default.jpg",
                face_id=None,
                s3_key=None
            )
        
        thumbnail_url = None
        
        if generate_url:
            try:
                s3_service = S3Service()
                
                # Adjust key based on size if needed
                s3_key = face.thumbnail_s3_key
                if size != "medium":
                    s3_key = s3_key.replace("_medium", f"_{size}")
                
                thumbnail_url = s3_service.generate_presigned_url(
                    s3_key,
                    expiration=3600
                )
                
            except S3ServiceException as e:
                logger.error(f"S3 error generating thumbnail URL: {str(e)}")
                thumbnail_url = "/static/avatars/default.jpg"
        
        return PersonThumbnailResponse(
            person_id=person_id,
            thumbnail_url=thumbnail_url,
            face_id=face.id,
            s3_key=face.thumbnail_s3_key
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get thumbnail for person {person_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve thumbnail"
        )


# =============================================================================
# Person Merging (With Background Tasks)
# =============================================================================

@router.post("/merge", response_model=PersonMergeResponse)
@limiter.limit("20/minute")
@require_roles(['photographer', 'admin'])
async def merge_persons(
    request: Request,
    merge_request: PersonMergeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Merge two persons into one (synchronous for small merges).
    
    **Security:**
    - Requires photographer or admin role
    - Verifies ownership of both persons
    - Rate limited (20 req/min)
    
    **Features:**
    - Transaction safety
    - Cache invalidation
    - Audit logging
    - Webhook notification
    
    **Note:** For merging >10 persons, use /batch-merge endpoint.
    """
    try:
        repo = PersonRepository(db)
        
        # Verify both persons exist
        source = repo.get_by_id(merge_request.source_person_id)
        target = repo.get_by_id(merge_request.target_person_id)
        
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source person not found"
            )
        if not target:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Target person not found"
            )
        
        # Verify same album
        if source.album_id != target.album_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot merge persons from different albums"
            )
        
        # Verify access
        await verify_album_ownership(source.album_id, current_user, db)
        
        # Count faces
        faces_count = repo.get_face_count(merge_request.source_person_id)
        
        # Perform merge
        merged_person = repo.merge_persons(
            merge_request.source_person_id,
            merge_request.target_person_id,
            merge_request.keep_source_name
        )
        
        db.commit()
        db.refresh(merged_person)
        
        # Invalidate cache
        await invalidate_person_cache(merge_request.source_person_id, source.album_id)
        await invalidate_person_cache(merge_request.target_person_id, target.album_id)
        await invalidate_cache(f"album:persons:{source.album_id}")
        
        # Audit log
        audit_log.record(
            user_id=current_user.id,
            action="person.merge",
            resource_type="person",
            resource_id=merged_person.id,
            details={
                "source_id": str(merge_request.source_person_id),
                "target_id": str(merge_request.target_person_id),
                "faces_transferred": faces_count
            }
        )
        
        # Webhook notification (background)
        background_tasks.add_task(
            trigger_webhook,
            event="person.merged",
            data={
                "merged_person_id": str(merged_person.id),
                "source_person_id": str(merge_request.source_person_id),
                "target_person_id": str(merge_request.target_person_id),
                "faces_transferred": faces_count
            }
        )
        
        logger.info(
            f"Merged person {merge_request.source_person_id} into {merge_request.target_person_id}",
            extra={"faces_transferred": faces_count}
        )
        
        return PersonMergeResponse(
            merged_person=PersonInDB.model_validate(merged_person),
            faces_transferred=faces_count,
            source_person_deleted=True
        )
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error during merge: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Merge operation failed"
        )


@router.get("/{person_id}/similar", response_model=SimilarPersonsResponse)
@limiter.limit("50/minute")
@cache_result(ttl=DEFAULT_CACHE_TTL)
async def find_similar_persons(
    request: Request,
    person_id: UUID = Path(..., description="Person ID"),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Find persons similar to this one (potential duplicates).
    
    **Uses:**
    - Face embedding similarity
    - Same album only
    - Quality weighted scoring
    
    **Features:**
    - Caching (5 min)
    - Rate limited (50 req/min)
    """
    try:
        repo = PersonRepository(db)
        person = repo.get_by_id(person_id)
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )
        
        # Verify access
        await verify_album_ownership(person.album_id, current_user, db)
        
        # Find similar persons
        similar = repo.find_similar_persons(person_id, threshold, limit)
        
        return SimilarPersonsResponse(
            person_id=person_id,
            similar_persons=similar,
            total_found=len(similar)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find similar persons: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find similar persons"
        )


@router.post("/{person_id}/split", response_model=PersonSplitResponse)
@limiter.limit("20/minute")
@require_roles(['photographer', 'admin'])
async def split_person(
    request: Request,
    person_id: UUID = Path(..., description="Person ID"),
    split_request: PersonSplitRequest = ...,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Split incorrectly merged person into two.
    
    **Security:**
    - Requires photographer or admin role
    - Verifies ownership
    - Rate limited (20 req/min)
    
    **Features:**
    - Validates face ownership
    - Transaction safety
    - Audit logging
    """
    try:
        repo = PersonRepository(db)
        person = repo.get_by_id(person_id)
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )
        
        # Verify access
        await verify_album_ownership(person.album_id, current_user, db)
        
        # Validate face IDs belong to this person
        person_faces = repo.get_faces(person_id)
        person_face_ids = {f.id for f in person_faces}
        
        invalid_faces = set(split_request.face_ids_to_split) - person_face_ids
        if invalid_faces:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid face IDs: {invalid_faces}"
            )
        
        # Check not splitting all faces
        if len(split_request.face_ids_to_split) >= len(person_face_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot split all faces. Delete person instead."
            )
        
        # Perform split
        result = repo.split_person(
            person_id,
            split_request.face_ids_to_split,
            split_request.new_person_name
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Split operation failed"
            )
        
        original, new_person = result
        db.commit()
        db.refresh(original)
        db.refresh(new_person)
        
        # Invalidate cache
        await invalidate_person_cache(person_id, person.album_id)
        await invalidate_cache(f"album:persons:{person.album_id}")
        
        # Audit log
        audit_log.record(
            user_id=current_user.id,
            action="person.split",
            resource_type="person",
            resource_id=person_id,
            details={
                "new_person_id": str(new_person.id),
                "faces_moved": len(split_request.face_ids_to_split)
            }
        )
        
        logger.info(f"Split person {person_id}, created {new_person.id}")
        
        return PersonSplitResponse(
            original_person=PersonInDB.model_validate(original),
            new_person=PersonInDB.model_validate(new_person),
            faces_moved=len(split_request.face_ids_to_split)
        )
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error during split: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Split operation failed"
        )


# =============================================================================
# Bulk Operations (Async Background Tasks)
# =============================================================================

@router.post("/batch-label", response_model=Dict[str, Any])
@limiter.limit("20/minute")
@require_roles(['photographer', 'admin'])
async def batch_label_faces(
    request: Request,
    label_request: EnhancedBatchLabelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Label multiple faces at once (async for large batches).
    
    **Security:**
    - Requires photographer or admin role
    - Verifies face ownership
    - Rate limited (20 req/min)
    
    **Performance:**
    - <10 faces: synchronous
    - ≥10 faces: background task
    
    **Returns:**
    - Immediate: result or job_id
    """
    try:
        # Determine if async needed
        is_async = len(label_request.face_ids) >= HEAVY_OPERATION_THRESHOLD
        
        repo = PersonRepository(db)
        
        # Handle person
        person = None
        created_new = False
        
        if label_request.person_id:
            person = repo.get_by_id(label_request.person_id)
            if not person:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Person not found"
                )
            
            # Verify access
            await verify_album_ownership(person.album_id, current_user, db)
            
        else:
            # Creating new person
            if not label_request.album_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="album_id required when creating new person"
                )
            
            # Verify album access
            await verify_album_ownership(label_request.album_id, current_user, db)
            
            # Create person
            person_data = PersonCreate(
                album_id=label_request.album_id,
                name=label_request.person_name or "Unknown Person"
            )
            person = repo.create(person_data)
            db.commit()
            db.refresh(person)
            created_new = True
        
        if is_async:
            # Enqueue background task
            task = batch_label_task.delay(
                face_ids=[str(fid) for fid in label_request.face_ids],
                person_id=str(person.id),
                user_id=str(current_user.id)
            )
            
            logger.info(f"Batch label job {task.id} started for {len(label_request.face_ids)} faces")
            
            return {
                "job_id": task.id,
                "status": "processing",
                "status_url": f"/api/v1/jobs/{task.id}",
                "person_id": str(person.id),
                "face_count": len(label_request.face_ids),
                "created_new_person": created_new
            }
        
        else:
            # Synchronous operation
            count = repo.batch_label_faces(label_request.face_ids, person.id)
            db.commit()
            
            # Invalidate cache
            await invalidate_person_cache(person.id, person.album_id)
            
            # Audit log
            audit_log.record(
                user_id=current_user.id,
                action="person.batch_label",
                resource_type="person",
                resource_id=person.id,
                details={
                    "faces_labeled": count,
                    "created_new": created_new
                }
            )
            
            logger.info(f"Batch labeled {count} faces to person {person.id}")
            
            return {
                "status": "completed",
                "person": PersonInDB.model_validate(person).dict(),
                "faces_labeled": count,
                "created_new_person": created_new
            }
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error during batch label: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch labeling failed"
        )


@router.post("/batch-merge", response_model=Dict[str, Any])
@limiter.limit("10/minute")
@require_roles(['photographer', 'admin'])
async def batch_merge_persons(
    request: Request,
    merge_request: EnhancedBatchMergeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Merge multiple persons into one (always async).
    
    **Security:**
    - Requires photographer or admin role
    - Verifies ownership of all persons
    - Rate limited (10 req/min)
    
    **Performance:**
    - Always runs as background task
    - Progress tracking available
    
    **Returns:**
    - Job ID for status tracking
    """
    try:
        repo = PersonRepository(db)
        
        # Determine target
        target_id = merge_request.target_person_id or merge_request.person_ids[0]
        
        if target_id not in merge_request.person_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Target person must be in person_ids list"
            )
        
        # Verify all persons exist and verify access
        persons = []
        for pid in merge_request.person_ids:
            person = repo.get_by_id(pid)
            if not person:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Person {pid} not found"
                )
            persons.append(person)
        
        # Check same album
        album_ids = {p.album_id for p in persons}
        if len(album_ids) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot merge persons from different albums"
            )
        
        # Verify access to album
        await verify_album_ownership(persons[0].album_id, current_user, db)
        
        # Enqueue background task
        task = batch_merge_task.delay(
            person_ids=[str(pid) for pid in merge_request.person_ids],
            target_id=str(target_id),
            merged_name=merge_request.merged_name,
            user_id=str(current_user.id)
        )
        
        # Audit log
        audit_log.record(
            user_id=current_user.id,
            action="person.batch_merge_initiated",
            details={
                "person_count": len(merge_request.person_ids),
                "job_id": task.id,
                "target_id": str(target_id)
            }
        )
        
        logger.info(
            f"Batch merge job {task.id} started",
            extra={"person_count": len(merge_request.person_ids)}
        )
        
        return {
            "job_id": task.id,
            "status": "processing",
            "status_url": f"/api/v1/jobs/{task.id}",
            "person_count": len(merge_request.person_ids),
            "target_person_id": str(target_id),
            "estimated_time_seconds": len(merge_request.person_ids) * 2
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start batch merge: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start merge operation"
        )


@router.post("/transfer", response_model=Dict[str, Any])
@limiter.limit("20/minute")
@require_roles(['photographer', 'admin'])
async def transfer_person(
    request: Request,
    transfer_request: PersonTransferRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Transfer person between albums (async for large persons).
    
    **Security:**
    - Requires photographer or admin role
    - Verifies access to both albums
    - Rate limited (20 req/min)
    
    **Performance:**
    - <50 faces: synchronous
    - ≥50 faces: background task
    """
    try:
        repo = PersonRepository(db)
        album_repo = AlbumRepository(db)
        
        # Verify person exists
        person = repo.get_by_id(transfer_request.person_id)
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Person not found"
            )
        
        # Verify source album access
        await verify_album_ownership(person.album_id, current_user, db)
        
        # Verify target album exists and access
        target_album = await verify_album_ownership(
            transfer_request.target_album_id,
            current_user,
            db
        )
        
        # Check if albums are the same
        if person.album_id == transfer_request.target_album_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Person is already in target album"
            )
        
        # Count faces
        face_count = repo.get_face_count(transfer_request.person_id)
        
        # Determine if async needed
        is_async = face_count >= 50
        
        if is_async:
            # Enqueue background task
            task = transfer_person_task.delay(
                person_id=str(transfer_request.person_id),
                target_album_id=str(transfer_request.target_album_id),
                merge_if_exists=transfer_request.merge_if_exists,
                user_id=str(current_user.id)
            )
            
            logger.info(f"Person transfer job {task.id} started")
            
            return {
                "job_id": task.id,
                "status": "processing",
                "status_url": f"/api/v1/jobs/{task.id}",
                "face_count": face_count
            }
        
        else:
            # Synchronous operation
            result = repo.transfer_to_album(
                transfer_request.person_id,
                transfer_request.target_album_id,
                transfer_request.merge_if_exists
            )
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Transfer failed"
                )
            
            transferred_person, was_merged, merged_with_id = result
            db.commit()
            db.refresh(transferred_person)
            
            # Invalidate cache
            await invalidate_person_cache(transfer_request.person_id, person.album_id)
            await invalidate_cache(f"album:persons:{person.album_id}")
            await invalidate_cache(f"album:persons:{transfer_request.target_album_id}")
            
            # Audit log
            audit_log.record(
                user_id=current_user.id,
                action="person.transfer",
                resource_type="person",
                resource_id=transfer_request.person_id,
                details={
                    "source_album": str(person.album_id),
                    "target_album": str(transfer_request.target_album_id),
                    "was_merged": was_merged,
                    "face_count": face_count
                }
            )
            
            logger.info(f"Transferred person {transfer_request.person_id}")
            
            return {
                "status": "completed",
                "person": PersonInDB.model_validate(transferred_person).dict(),
                "faces_transferred": face_count,
                "merged_with_existing": was_merged,
                "merged_person_id": str(merged_with_id) if merged_with_id else None
            }
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error during transfer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transfer operation failed"
        )


# =============================================================================
# Health Check
# =============================================================================

@router.get("/health")
async def health_check():
    """
    Health check for person service.
    
    **Returns:**
    - Service status
    - Cache connectivity
    - Database connectivity
    """
    health_status = {
        "status": "healthy",
        "service": "person-api",
        "timestamp": "2025-01-01T00:00:00Z"
    }
    
    # Check cache
    try:
        redis_client.ping()
        health_status["cache"] = "healthy"
    except Exception as e:
        health_status["cache"] = "unhealthy"
        health_status["cache_error"] = str(e)
        health_status["status"] = "degraded"
    
    # Check database
    try:
        from src.db.session import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        health_status["database"] = "healthy"
    except Exception as e:
        health_status["database"] = "unhealthy"
        health_status["database_error"] = str(e)
        health_status["status"] = "unhealthy"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    
    return JSONResponse(
        status_code=status_code,
        content=health_status
    )


# =============================================================================
# Error Handlers
# =============================================================================

# Note: Exception handlers should be registered at the app level in main.py
# not at the router level. These are included here for reference only.
#
# Example usage in main.py:
# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc: HTTPException):
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"error": exc.detail, "status_code": exc.status_code}
#     )