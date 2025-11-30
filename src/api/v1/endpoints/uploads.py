"""Upload API endpoints for photo uploads."""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
from datetime import datetime, timedelta

from src.api.deps import get_db, get_current_user
from src.models.user import User
from src.models.enums import PhotoStatus
from src.repositories.photo_repo import PhotoRepository
from src.repositories.album_repo import AlbumRepository
from src.schemas.upload import (
    PresignRequest,
    PresignResponse,
    BulkPresignRequest,
    BulkPresignResponse,
    UploadCompleteRequest,
    UploadCompleteResponse,
    BulkUploadCompleteRequest,
    BulkUploadCompleteResponse,
    MultipartUploadInitRequest,
    MultipartUploadInitResponse,
    MultipartPartPresignRequest,
    MultipartPartPresignResponse,
    MultipartUploadCompleteRequest,
    MultipartUploadAbortRequest,
    UploadProgressRequest,
    UploadProgressResponse,
    UploadStatsResponse,
    UploadQuotaResponse
)
from src.services.storage.s3 import S3Service
from src.app.config import settings


router = APIRouter()


def check_album_access(album_id: UUID, user: User, db: Session) -> None:
    """
    Verify user has upload access to album.
    
    Args:
        album_id: Album UUID
        user: Current user
        db: Database session
        
    Raises:
        HTTPException if album not found or no access
    """
    album_repo = AlbumRepository(db)
    album = album_repo.get(album_id)
    
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    # Check permission (owner or admin)
    if user.role != 'admin' and album.photographer_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized to upload to this album'
        )


def check_upload_quota(album_id: UUID, filesize: int, db: Session) -> None:
    """
    Check if upload quota is available.
    
    Args:
        album_id: Album UUID
        filesize: File size to upload
        db: Database session
        
    Raises:
        HTTPException if quota exceeded
    """
    photo_repo = PhotoRepository(db)
    
    # TODO: Implement actual quota checking based on subscription plan
    # For now, just check basic limits
    
    # Example: Max 10,000 photos per album
    stats = photo_repo.get_album_stats(album_id)
    if stats['total_photos'] >= 10000:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Album photo limit reached (10,000 photos)'
        )
    
    # Example: Max 100GB per album
    max_storage = 100 * 1024 * 1024 * 1024  # 100GB
    if stats['total_size_bytes'] + filesize > max_storage:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Album storage limit reached (100GB)'
        )


@router.post('/albums/{album_id}/presign', response_model=PresignResponse)
def request_presigned_url(
    album_id: UUID,
    request: PresignRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Request presigned URL for photo upload (Step 1 of upload flow).
    
    Flow:
    1. Client requests presigned URL
    2. Server creates photo record and returns S3 presigned URL
    3. Client uploads directly to S3
    4. Client notifies completion via /notify_upload
    
    Features:
    - Direct S3 upload (no file passes through server)
    - Supports files up to 100MB (use multipart for larger)
    - Returns photo_id for tracking
    """
    # Verify access
    check_album_access(album_id, current_user, db)
    
    # Check quota
    check_upload_quota(album_id, request.filesize, db)
    
    # Initialize services
    s3_service = S3Service()
    photo_repo = PhotoRepository(db)
    
    # Generate S3 key
    s3_key = s3_service.generate_s3_key(str(album_id), request.filename)
    
    # Create photo record in database
    photo = photo_repo.create_photo(
        album_id=album_id,
        uploader_id=current_user.id,
        s3_key=s3_key,
        s3_bucket=s3_service.bucket_name,
        filename=request.filename,
        content_type=request.content_type,
        filesize=request.filesize
    )
    
    # Generate presigned URL
    presign_data = s3_service.generate_presigned_upload_url(
        s3_key=s3_key,
        content_type=request.content_type,
        expires_in=3600,
        use_multipart=False
    )
    
    return PresignResponse(
        upload_url=presign_data['upload_url'],
        photo_id=photo.id,
        s3_key=s3_key,
        s3_bucket=s3_service.bucket_name,
        fields=presign_data.get('fields'),
        expires_in=3600,
        method=presign_data['method']
    )


@router.post('/albums/{album_id}/presign/bulk', response_model=BulkPresignResponse)
def request_bulk_presigned_urls(
    album_id: UUID,
    request: BulkPresignRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Request multiple presigned URLs for bulk upload.
    
    Features:
    - Upload up to 100 files at once
    - Efficient batch processing
    - Returns all presigned URLs in single response
    
    Max 100 files per request.
    """
    # Verify access
    check_album_access(album_id, current_user, db)
    
    # Check total quota
    total_size = sum(f.filesize for f in request.files)
    check_upload_quota(album_id, total_size, db)
    
    # Initialize services
    s3_service = S3Service()
    photo_repo = PhotoRepository(db)
    
    uploads = []
    for file_request in request.files:
        # Generate S3 key
        s3_key = s3_service.generate_s3_key(str(album_id), file_request.filename)
        
        # Create photo record
        photo = photo_repo.create_photo(
            album_id=album_id,
            uploader_id=current_user.id,
            s3_key=s3_key,
            s3_bucket=s3_service.bucket_name,
            filename=file_request.filename,
            content_type=file_request.content_type,
            filesize=file_request.filesize
        )
        
        # Generate presigned URL
        presign_data = s3_service.generate_presigned_upload_url(
            s3_key=s3_key,
            content_type=file_request.content_type,
            expires_in=3600
        )
        
        uploads.append(PresignResponse(
            upload_url=presign_data['upload_url'],
            photo_id=photo.id,
            s3_key=s3_key,
            s3_bucket=s3_service.bucket_name,
            fields=presign_data.get('fields'),
            expires_in=3600,
            method=presign_data['method']
        ))
    
    return BulkPresignResponse(
        uploads=uploads,
        total=len(uploads),
        estimated_total_size=total_size
    )


@router.post('/albums/{album_id}/notify_upload', response_model=UploadCompleteResponse)
def notify_upload_complete(
    album_id: UUID,
    request: UploadCompleteRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Notify that upload is complete (Step 3 of upload flow).
    
    After client uploads to S3:
    1. Client calls this endpoint with photo_id and s3_key
    2. Server updates photo record
    3. Server enqueues background processing (thumbnails, faces)
    
    Processing includes:
    - EXIF extraction
    - Thumbnail generation (small, medium, large)
    - Face detection
    - Watermark application
    """
    # Verify access
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    s3_service = S3Service()
    
    # Get photo record
    photo = photo_repo.get(request.photo_id)
    if not photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Photo not found'
        )
    
    # Verify photo belongs to this album
    if photo.album_id != album_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Photo does not belong to this album'
        )
    
    # Verify S3 key matches
    if photo.s3_key != request.s3_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='S3 key mismatch'
        )
    
    # Verify file exists in S3
    if not s3_service.check_object_exists(request.s3_key):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='File not found in S3. Upload may have failed.'
        )
    
    # Update EXIF if provided
    if request.exif:
        photo_repo.update_exif(
            photo.id,
            exif_data=request.exif,
            taken_at=request.exif.get('taken_at'),
            camera_model=request.exif.get('camera_model')
        )
    
    # Update status to processing
    photo_repo.update_status(photo.id, PhotoStatus.processing)
    
    # Enqueue background processing
    # TODO: Implement Celery task
    # from src.tasks.workers.photo_processor import process_photo
    # background_tasks.add_task(process_photo.delay, str(photo.id))
    
    return UploadCompleteResponse(
        photo_id=photo.id,
        status='processing',
        message='Upload complete. Processing started.',
        processing_started=True
    )


@router.post('/albums/{album_id}/notify_upload/bulk', response_model=BulkUploadCompleteResponse)
def notify_bulk_upload_complete(
    album_id: UUID,
    request: BulkUploadCompleteRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Notify completion of multiple uploads.
    
    Efficient batch notification for bulk uploads.
    Max 100 photos per request.
    """
    # Verify access
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    s3_service = S3Service()
    
    success_count = 0
    failed_photo_ids = []
    
    for upload in request.uploads:
        try:
            # Get photo
            photo = photo_repo.get(upload.photo_id)
            if not photo or photo.album_id != album_id:
                failed_photo_ids.append(upload.photo_id)
                continue
            
            # Verify S3 upload
            if not s3_service.check_object_exists(upload.s3_key):
                failed_photo_ids.append(upload.photo_id)
                continue
            
            # Update EXIF if provided
            if upload.exif:
                photo_repo.update_exif(upload.photo_id, upload.exif)
            
            # Update status
            photo_repo.update_status(upload.photo_id, PhotoStatus.processing)
            
            # Enqueue processing
            # TODO: background_tasks.add_task(process_photo.delay, str(upload.photo_id))
            
            success_count += 1
            
        except Exception as e:
            failed_photo_ids.append(upload.photo_id)
    
    return BulkUploadCompleteResponse(
        success_count=success_count,
        failed_count=len(failed_photo_ids),
        failed_photo_ids=failed_photo_ids,
        message=f'Processed {success_count}/{len(request.uploads)} uploads'
    )


@router.post('/albums/{album_id}/multipart/init', response_model=MultipartUploadInitResponse)
def initialize_multipart_upload(
    album_id: UUID,
    request: MultipartUploadInitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Initialize multipart upload for large files (>100MB).
    
    Multipart upload flow:
    1. Initialize multipart upload (this endpoint)
    2. Get presigned URLs for each part
    3. Upload parts to S3
    4. Complete multipart upload
    
    Required for files > 5MB.
    Supports files up to 5TB (AWS limit).
    """
    # Verify access
    check_album_access(album_id, current_user, db)
    check_upload_quota(album_id, request.filesize, db)
    
    s3_service = S3Service()
    photo_repo = PhotoRepository(db)
    
    # Generate S3 key
    s3_key = s3_service.generate_s3_key(str(album_id), request.filename)
    
    # Create photo record
    photo = photo_repo.create_photo(
        album_id=album_id,
        uploader_id=current_user.id,
        s3_key=s3_key,
        s3_bucket=s3_service.bucket_name,
        filename=request.filename,
        content_type=request.content_type,
        filesize=request.filesize
    )
    
    # Calculate parts
    total_parts, actual_part_size = s3_service.calculate_multipart_info(
        request.filesize,
        request.part_size
    )
    
    # Initialize multipart upload in S3
    upload_id = s3_service.initiate_multipart_upload(
        s3_key=s3_key,
        content_type=request.content_type
    )
    
    # Store upload_id in photo extra_data for tracking
    photo_repo.update(photo.id, {
        'extra_data': {
            'multipart_upload_id': upload_id,
            'total_parts': total_parts,
            'part_size': actual_part_size
        }
    })
    
    return MultipartUploadInitResponse(
        photo_id=photo.id,
        upload_id=upload_id,
        s3_key=s3_key,
        s3_bucket=s3_service.bucket_name,
        total_parts=total_parts,
        part_size=actual_part_size
    )


@router.post('/multipart/parts', response_model=MultipartPartPresignResponse)
def get_multipart_part_urls(
    request: MultipartPartPresignRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get presigned URLs for multipart upload parts.
    
    Request URLs for specific part numbers.
    Can request multiple parts at once (up to 100).
    
    Client should upload each part using the returned URLs.
    """
    photo_repo = PhotoRepository(db)
    
    # Verify photo exists and user has access
    photo = photo_repo.get(request.photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    if not photo_repo.verify_album_ownership(request.photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
    # Generate presigned URLs for parts
    s3_service = S3Service()
    part_urls = s3_service.generate_multipart_presigned_urls(
        s3_key=request.s3_key,
        upload_id=request.upload_id,
        part_numbers=request.part_numbers,
        expires_in=3600
    )
    
    return MultipartPartPresignResponse(
        part_urls=part_urls,
        expires_in=3600
    )


@router.post('/multipart/complete', response_model=UploadCompleteResponse)
def complete_multipart_upload(
    request: MultipartUploadCompleteRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Complete multipart upload.
    
    After all parts are uploaded:
    1. Client calls this endpoint with part ETags
    2. Server completes multipart upload in S3
    3. Server enqueues processing
    
    Parts must be provided in order with ETags from S3 responses.
    """
    photo_repo = PhotoRepository(db)
    
    # Verify photo and access
    photo = photo_repo.get(request.photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    if not photo_repo.verify_album_ownership(request.photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
    # Complete multipart upload in S3
    s3_service = S3Service()
    
    # Format parts for S3
    parts = [
        {'PartNumber': part['part_number'], 'ETag': part['etag']}
        for part in request.parts
    ]
    
    try:
        result = s3_service.complete_multipart_upload(
            s3_key=request.s3_key,
            upload_id=request.upload_id,
            parts=parts
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Failed to complete multipart upload: {str(e)}'
        )
    
    # Update EXIF if provided
    if request.exif:
        photo_repo.update_exif(request.photo_id, request.exif)
    
    # Update status to processing
    photo_repo.update_status(request.photo_id, PhotoStatus.processing)
    
    # Enqueue processing
    # TODO: background_tasks.add_task(process_photo.delay, str(request.photo_id))
    
    return UploadCompleteResponse(
        photo_id=request.photo_id,
        status='processing',
        message='Multipart upload complete. Processing started.',
        processing_started=True
    )


@router.post('/multipart/abort', status_code=status.HTTP_204_NO_CONTENT)
def abort_multipart_upload(
    request: MultipartUploadAbortRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Abort multipart upload and cleanup.
    
    Cancels the upload and deletes all uploaded parts from S3.
    Also deletes the photo record.
    """
    photo_repo = PhotoRepository(db)
    
    # Verify photo and access
    photo = photo_repo.get(request.photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    if not photo_repo.verify_album_ownership(request.photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
    # Abort multipart upload in S3
    s3_service = S3Service()
    s3_service.abort_multipart_upload(
        s3_key=request.s3_key,
        upload_id=request.upload_id
    )
    
    # Delete photo record
    photo_repo.delete(request.photo_id, soft=False)
    
    return None


@router.post('/progress', response_model=UploadProgressResponse)
def update_upload_progress(
    request: UploadProgressRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update upload progress (for monitoring and UI).
    
    Optional endpoint for clients to report upload progress.
    Useful for:
    - Progress bars
    - Upload monitoring
    - Analytics
    """
    photo_repo = PhotoRepository(db)
    
    # Verify photo exists
    photo = photo_repo.get(request.photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    # Calculate percentage
    percentage = (request.bytes_uploaded / request.total_bytes) * 100
    
    # Update extra_data with progress
    photo_repo.update(request.photo_id, {
        'extra_data': {
            'upload_progress': {
                'bytes_uploaded': request.bytes_uploaded,
                'total_bytes': request.total_bytes,
                'percentage': round(percentage, 2),
                'last_updated': datetime.utcnow().isoformat()
            }
        }
    })
    
    return UploadProgressResponse(
        photo_id=request.photo_id,
        bytes_uploaded=request.bytes_uploaded,
        total_bytes=request.total_bytes,
        percentage=round(percentage, 2),
        status='uploading' if percentage < 100 else 'complete'
    )


@router.get('/albums/{album_id}/stats', response_model=UploadStatsResponse)
def get_upload_stats(
    album_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get upload statistics for monitoring.
    
    Returns:
    - Total uploads today
    - Successful vs failed
    - Total bytes uploaded
    - Pending uploads
    """
    # Verify access
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    
    # Get today's uploads
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    query = photo_repo.db.query(photo_repo.model).filter(
        photo_repo.model.album_id == album_id,
        photo_repo.model.created_at >= today_start
    )
    
    total_today = query.count()
    successful_today = query.filter(photo_repo.model.status == PhotoStatus.done).count()
    failed_today = query.filter(photo_repo.model.status == PhotoStatus.failed).count()
    
    total_bytes = photo_repo.db.query(
        photo_repo.db.func.coalesce(photo_repo.db.func.sum(photo_repo.model.filesize), 0)
    ).filter(
        photo_repo.model.album_id == album_id,
        photo_repo.model.created_at >= today_start
    ).scalar() or 0
    
    pending = query.filter(photo_repo.model.status == PhotoStatus.uploaded).count()
    
    return UploadStatsResponse(
        album_id=album_id,
        total_uploads_today=total_today,
        successful_uploads_today=successful_today,
        failed_uploads_today=failed_today,
        total_bytes_uploaded_today=int(total_bytes),
        pending_uploads=pending
    )


@router.get('/albums/{album_id}/quota', response_model=UploadQuotaResponse)
def get_upload_quota(
    album_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get upload quota information.
    
    Returns current usage and remaining quota.
    Useful for showing limits in UI before upload.
    """
    # Verify access
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    stats = photo_repo.get_album_stats(album_id)
    
    # TODO: Get actual limits from subscription/plan
    max_photos = 10000  # Example limit
    max_storage = 100 * 1024 * 1024 * 1024  # 100GB
    
    current_photos = stats['total_photos']
    current_storage = stats['total_size_bytes']
    
    remaining_photos = max_photos - current_photos
    remaining_storage = max_storage - current_storage
    
    can_upload = remaining_photos > 0 and remaining_storage > 0
    reason = None
    
    if remaining_photos <= 0:
        reason = 'Photo limit reached'
    elif remaining_storage <= 0:
        reason = 'Storage limit reached'
    
    return UploadQuotaResponse(
        album_id=album_id,
        max_photos=max_photos,
        current_photos=current_photos,
        remaining_photos=remaining_photos,
        max_storage_bytes=max_storage,
        current_storage_bytes=current_storage,
        remaining_storage_bytes=remaining_storage,
        can_upload=can_upload,
        reason=reason
    )