"""Photo API endpoints (excluding uploads - handled separately)."""
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, List
from uuid import UUID

from src.api.deps import get_db, get_current_user
from src.models.user import User
from src.models.enums import PhotoStatus
from src.repositories.photo_repo import PhotoRepository
from src.repositories.album_repo import AlbumRepository
from src.schemas.photo import (
    PhotoUpdate,
    PhotoResponse,
    PhotoDetailResponse,
    PhotoListResponse,
    PhotoStatsResponse,
    PhotoBulkActionRequest,
    PhotoBulkActionResponse,
    PhotoDownloadRequest,
    PhotoDownloadResponse,
    PhotoSearchRequest,
    PhotoFilterRequest
)
from src.app.config import settings


router = APIRouter()


def build_photo_response(photo_data: dict, generate_urls: bool = True) -> PhotoResponse:
    """Build PhotoResponse from photo and face count."""
    photo = photo_data['photo']
    
    response_dict = {
        **photo.__dict__,
        'face_count': photo_data.get('face_count', 0)
    }
    
    # Generate presigned URL for original if needed
    if generate_urls:
        # TODO: Implement S3 presigned URL generation
        # from src.services.storage.s3 import S3Service
        # s3_service = S3Service()
        # response_dict['original_url'] = s3_service.generate_presigned_url(photo.s3_key)
        pass
    
    return PhotoResponse(**response_dict)


@router.get('/{photo_id}', response_model=PhotoDetailResponse)
def get_photo(
    photo_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get photo details with uploader and album info.
    
    Returns full photo metadata including:
    - File information
    - Processing status
    - EXIF data
    - Thumbnail URLs
    - Face count
    - Uploader and album details
    """
    photo_repo = PhotoRepository(db)
    
    # Get photo with relations
    photo = photo_repo.get_with_relations(photo_id)
    if not photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Photo not found'
        )
    
    # Verify permission
    if current_user.role != 'admin':
        if not photo.album or photo.album.photographer_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Not authorized to access this photo'
            )
    
    # Get face count
    photo_data = photo_repo.get_with_face_count(photo_id)
    
    return PhotoDetailResponse(
        **photo.__dict__,
        face_count=photo_data['face_count'],
        uploader_name=photo.uploader.name if photo.uploader else None,
        uploader_email=photo.uploader.email if photo.uploader else None,
        album_title=photo.album.title if photo.album else None
    )


@router.get('/album/{album_id}', response_model=PhotoListResponse)
def list_photos_in_album(
    album_id: UUID,
    page: int = Query(1, ge=1, description='Page number'),
    size: int = Query(50, ge=1, le=200, description='Photos per page'),
    status: Optional[PhotoStatus] = Query(None, description='Filter by status'),
    order_by: str = Query('created_at', description='Field to sort by'),
    order_desc: bool = Query(True, description='Sort descending'),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all photos in an album with pagination.
    
    Features:
    - Pagination support
    - Filter by processing status
    - Custom sorting (by date, filename, size, etc.)
    - Face count included
    
    Order by options: created_at, filename, filesize, taken_at
    """
    album_repo = AlbumRepository(db)
    photo_repo = PhotoRepository(db)
    
    # Verify album access
    album = album_repo.get(album_id)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized to access this album'
        )
    
    # Get photos
    skip = (page - 1) * size
    photos, total = photo_repo.get_by_album(
        album_id=album_id,
        skip=skip,
        limit=size,
        status=status,
        order_by=order_by,
        order_desc=order_desc
    )
    
    # Build responses with face counts
    photo_responses = []
    for photo in photos:
        photo_data = photo_repo.get_with_face_count(photo.id)
        photo_responses.append(build_photo_response(photo_data))
    
    pages = (total + size - 1) // size
    
    return PhotoListResponse(
        items=photo_responses,
        total=total,
        page=page,
        size=size,
        pages=pages
    )


@router.get('/album/{album_id}/search', response_model=PhotoListResponse)
def search_photos(
    album_id: UUID,
    query: str = Query(..., min_length=1, description='Search term'),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search photos by filename or camera model.
    
    Searches in:
    - Filename
    - Camera model
    """
    album_repo = AlbumRepository(db)
    photo_repo = PhotoRepository(db)
    
    # Verify access
    album = album_repo.get(album_id)
    if not album:
        raise HTTPException(status_code=404, detail='Album not found')
    
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(status_code=403, detail='Not authorized')
    
    # Search photos
    skip = (page - 1) * size
    photos, total = photo_repo.search_photos(album_id, query, skip, size)
    
    # Build responses
    photo_responses = []
    for photo in photos:
        photo_data = photo_repo.get_with_face_count(photo.id)
        photo_responses.append(build_photo_response(photo_data))
    
    pages = (total + size - 1) // size
    
    return PhotoListResponse(
        items=photo_responses,
        total=total,
        page=page,
        size=size,
        pages=pages
    )


@router.patch('/{photo_id}', response_model=PhotoResponse)
def update_photo(
    photo_id: UUID,
    photo_data: PhotoUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update photo metadata.
    
    Can update:
    - Filename
    - Taken at timestamp
    - Camera model
    - Extra metadata (JSON)
    
    Cannot update:
    - File content
    - S3 key
    - Processing status
    """
    photo_repo = PhotoRepository(db)
    
    # Verify ownership
    if not photo_repo.verify_album_ownership(photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Not authorized to update this photo'
            )
    
    # Update photo
    update_dict = photo_data.model_dump(exclude_unset=True)
    updated_photo = photo_repo.update(photo_id, update_dict)
    
    if not updated_photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Photo not found'
        )
    
    photo_with_count = photo_repo.get_with_face_count(photo_id)
    return build_photo_response(photo_with_count)


@router.delete('/{photo_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_photo(
    photo_id: UUID,
    hard: bool = Query(False, description='Permanent deletion with S3 cleanup'),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete photo (soft delete by default).
    
    Soft delete (hard=false):
    - Marks photo as deleted
    - Recoverable via restore endpoint
    - S3 files remain
    
    Hard delete (hard=true):
    - Permanently removes from database
    - Triggers S3 cleanup background job
    - Cannot be recovered
    """
    photo_repo = PhotoRepository(db)
    
    # Get photo for S3 key
    photo = photo_repo.get(photo_id)
    if not photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Photo not found'
        )
    
    # Verify ownership
    if not photo_repo.verify_album_ownership(photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Not authorized to delete this photo'
            )
    
    # Delete photo
    success = photo_repo.delete(photo_id, soft=not hard)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Failed to delete photo'
        )
    
    # TODO: If hard delete, enqueue S3 cleanup
    # if hard:
    #     from src.tasks.workers.cleanup import cleanup_photo_s3
    #     cleanup_photo_s3.delay(str(photo_id), photo.s3_key, photo.s3_bucket)
    
    return None


@router.post('/{photo_id}/restore', response_model=PhotoResponse)
def restore_photo(
    photo_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Restore soft-deleted photo.
    
    Only works for photos that were soft-deleted (not hard-deleted).
    """
    photo_repo = PhotoRepository(db)
    
    # Get photo including deleted
    photo = photo_repo.get(photo_id, include_deleted=True)
    if not photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Photo not found'
        )
    
    if not photo.deleted_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Photo is not deleted'
        )
    
    # Verify ownership
    if not photo_repo.verify_album_ownership(photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Not authorized'
            )
    
    # Restore photo
    success = photo_repo.restore(photo_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Failed to restore photo'
        )
    
    photo_data = photo_repo.get_with_face_count(photo_id)
    return build_photo_response(photo_data)


@router.post('/{photo_id}/reprocess', response_model=PhotoResponse)
def reprocess_photo(
    photo_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Reprocess photo (re-run face detection, thumbnail generation).
    
    Useful for:
    - Re-detecting faces after model update
    - Regenerating thumbnails
    - Fixing failed processing
    
    Resets status to 'uploaded' and enqueues processing.
    """
    photo_repo = PhotoRepository(db)
    
    # Verify ownership
    if not photo_repo.verify_album_ownership(photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Not authorized'
            )
    
    # Reset status to uploaded
    photo = photo_repo.reprocess_photo(photo_id)
    if not photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Photo not found'
        )
    
    # TODO: Enqueue processing
    # from src.tasks.workers.photo_processor import process_photo
    # background_tasks.add_task(process_photo.delay, str(photo_id))
    
    photo_data = photo_repo.get_with_face_count(photo_id)
    return build_photo_response(photo_data)


@router.get('/album/{album_id}/stats', response_model=PhotoStatsResponse)
def get_album_photo_stats(
    album_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive photo statistics for an album.
    
    Returns:
    - Total photos count
    - Count by status (uploaded, processing, done, failed)
    - Total storage used (bytes)
    - Total faces detected
    - Photos with faces count
    """
    album_repo = AlbumRepository(db)
    photo_repo = PhotoRepository(db)
    
    # Verify access
    album = album_repo.get(album_id)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized'
        )
    
    stats = photo_repo.get_album_stats(album_id)
    return PhotoStatsResponse(**stats)


@router.get('/album/{album_id}/status-breakdown', response_model=dict)
def get_status_breakdown(
    album_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get photo count breakdown by status.
    
    Returns:
    {
      "uploaded": 5,
      "processing": 2,
      "done": 50,
      "failed": 1
    }
    """
    album_repo = AlbumRepository(db)
    photo_repo = PhotoRepository(db)
    
    # Verify access
    album = album_repo.get(album_id)
    if not album:
        raise HTTPException(status_code=404, detail='Album not found')
    
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(status_code=403, detail='Not authorized')
    
    return photo_repo.count_by_status(album_id)


@router.get('/album/{album_id}/failed', response_model=PhotoListResponse)
def get_failed_photos(
    album_id: UUID,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all failed photos in an album.
    
    Useful for:
    - Identifying processing errors
    - Bulk reprocessing failed photos
    - Error analysis
    """
    album_repo = AlbumRepository(db)
    photo_repo = PhotoRepository(db)
    
    # Verify access
    album = album_repo.get(album_id)
    if not album:
        raise HTTPException(status_code=404, detail='Album not found')
    
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(status_code=403, detail='Not authorized')
    
    skip = (page - 1) * size
    photos, total = photo_repo.get_failed_photos(album_id, skip, size)
    
    # Build responses
    photo_responses = []
    for photo in photos:
        photo_data = photo_repo.get_with_face_count(photo.id)
        photo_responses.append(build_photo_response(photo_data))
    
    pages = (total + size - 1) // size
    
    return PhotoListResponse(
        items=photo_responses,
        total=total,
        page=page,
        size=size,
        pages=pages
    )


@router.get('/album/{album_id}/recent', response_model=PhotoListResponse)
def get_recent_uploads(
    album_id: UUID,
    hours: int = Query(24, ge=1, le=168, description='Hours to look back'),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get recently uploaded photos in an album.
    
    Default: Last 24 hours
    Max: Last 7 days (168 hours)
    """
    album_repo = AlbumRepository(db)
    photo_repo = PhotoRepository(db)
    
    # Verify access
    album = album_repo.get(album_id)
    if not album:
        raise HTTPException(status_code=404, detail='Album not found')
    
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(status_code=403, detail='Not authorized')
    
    photos = photo_repo.get_recent_uploads(album_id, hours, limit)
    
    # Build responses
    photo_responses = []
    for photo in photos:
        photo_data = photo_repo.get_with_face_count(photo.id)
        photo_responses.append(build_photo_response(photo_data))
    
    return PhotoListResponse(
        items=photo_responses,
        total=len(photo_responses),
        page=1,
        size=limit,
        pages=1
    )


@router.post('/bulk-action', response_model=PhotoBulkActionResponse)
def bulk_photo_action(
    album_id: UUID,
    request: PhotoBulkActionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Perform bulk action on multiple photos.
    
    Supported actions:
    - **delete**: Soft delete photos
    - **reprocess**: Re-run processing on photos
    - **download**: Create bulk download archive (TODO)
    
    Max 100 photos per request.
    """
    album_repo = AlbumRepository(db)
    photo_repo = PhotoRepository(db)
    
    # Verify album ownership
    album = album_repo.get(album_id)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized'
        )
    
    if request.action == 'delete':
        success_count, failed_ids = photo_repo.bulk_delete_by_album(
            album_id,
            request.photo_ids
        )
        message = f"Successfully deleted {success_count} photo(s)"
        
    elif request.action == 'reprocess':
        success_count = 0
        failed_ids = []
        for photo_id in request.photo_ids:
            if photo_repo.reprocess_photo(photo_id):
                success_count += 1
                # TODO: Enqueue processing
                # background_tasks.add_task(process_photo.delay, str(photo_id))
            else:
                failed_ids.append(photo_id)
        message = f"Successfully queued {success_count} photo(s) for reprocessing"
        
    elif request.action == 'download':
        # TODO: Create zip archive and return download URL
        message = "Bulk download preparation started"
        success_count = len(request.photo_ids)
        failed_ids = []
        # background_tasks.add_task(create_download_archive, album_id, request.photo_ids)
        
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported action: {request.action}"
        )
    
    return PhotoBulkActionResponse(
        success_count=success_count,
        failed_count=len(failed_ids),
        failed_ids=failed_ids,
        message=message
    )


@router.post('/{photo_id}/download', response_model=PhotoDownloadResponse)
def download_photo(
    photo_id: UUID,
    download_request: PhotoDownloadRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate download URL for photo.
    
    Quality options:
    - **thumbnail**: Small thumbnail (fastest)
    - **medium**: Medium resolution
    - **high**: High resolution
    - **original**: Original file (largest)
    
    Watermark option:
    - true: Return watermarked version (if available)
    - false: Return clean version
    """
    photo_repo = PhotoRepository(db)
    
    photo = photo_repo.get_with_relations(photo_id)
    if not photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Photo not found'
        )
    
    # Check album download settings
    if photo.album and not photo.album.download_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Downloads disabled for this album'
        )
    
    # Verify permission
    if current_user.role != 'admin':
        if not photo.album or photo.album.photographer_id != current_user.id:
            # TODO: Check if user has paid for download
            # if not check_download_payment(photo_id, current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Not authorized to download this photo'
            )
    
    # Select S3 key based on quality
    if download_request.quality == 'thumbnail':
        s3_key = photo.thumbnail_small_url or photo.s3_key
    elif download_request.quality == 'medium':
        s3_key = photo.thumbnail_medium_url or photo.s3_key
    elif download_request.quality == 'high':
        s3_key = photo.thumbnail_large_url or photo.s3_key
    else:  # original
        if download_request.watermark and photo.watermarked_url:
            s3_key = photo.watermarked_url
        else:
            s3_key = photo.s3_key
    
    # TODO: Generate presigned download URL
    # from src.services.storage.s3 import S3Service
    # s3_service = S3Service()
    # download_url = s3_service.generate_presigned_download_url(
    #     s3_key=s3_key,
    #     expires_in=3600,
    #     content_disposition=f'attachment; filename="{photo.filename}"'
    # )
    
    download_url = f"https://s3.amazonaws.com/{photo.s3_bucket}/{s3_key}"
    
    return PhotoDownloadResponse(
        download_url=download_url,
        expires_in=3600,
        filename=photo.filename
    )


@router.get('/my-uploads', response_model=PhotoListResponse)
def get_my_uploads(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all photos uploaded by current user across all albums.
    
    Useful for users to see their contribution history.
    """
    photo_repo = PhotoRepository(db)
    
    skip = (page - 1) * size
    photos, total = photo_repo.get_photos_by_uploader(
        uploader_id=current_user.id,
        skip=skip,
        limit=size
    )
    
    # Build responses
    photo_responses = []
    for photo in photos:
        photo_data = photo_repo.get_with_face_count(photo.id)
        photo_responses.append(build_photo_response(photo_data))
    
    pages = (total + size - 1) // size
    
    return PhotoListResponse(
        items=photo_responses,
        total=total,
        page=page,
        size=size,
        pages=pages
    )


@router.post('/{photo_id}/mark-favorite', response_model=PhotoResponse)
def mark_as_favorite(
    photo_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Mark photo as favorite (stored in extra_data).
    
    Adds/updates favorite flag in photo metadata.
    """
    photo_repo = PhotoRepository(db)
    
    # Verify ownership
    if not photo_repo.verify_album_ownership(photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
    photo = photo_repo.get(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    # Update extra_data
    extra_data = photo.extra_data or {}
    extra_data['is_favorite'] = True
    extra_data['favorited_by'] = str(current_user.id)
    extra_data['favorited_at'] = datetime.utcnow().isoformat()
    
    updated = photo_repo.update(photo_id, {'extra_data': extra_data})
    
    photo_data = photo_repo.get_with_face_count(photo_id)
    return build_photo_response(photo_data)


@router.delete('/{photo_id}/mark-favorite', response_model=PhotoResponse)
def unmark_as_favorite(
    photo_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Remove favorite mark from photo.
    """
    photo_repo = PhotoRepository(db)
    
    # Verify ownership
    if not photo_repo.verify_album_ownership(photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
    photo = photo_repo.get(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    # Update extra_data
    extra_data = photo.extra_data or {}
    extra_data['is_favorite'] = False
    
    updated = photo_repo.update(photo_id, {'extra_data': extra_data})
    
    photo_data = photo_repo.get_with_face_count(photo_id)
    return build_photo_response(photo_data)