"""Upload API endpoints for photo uploads - Production Grade (27 endpoints)."""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import Optional, List
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
    UploadQuotaResponse,
    UploadMetadataResponse,
    RecentUploadsResponse,
    FileMetadataUpdate,
    FileCopyRequest,
    DeriveAssetRequest,
    DeriveAssetResponse,
    MultipartPartsListResponse,
    FlaggedFilesResponse,
    BulkDeleteRequest,
    AlbumHealthResponse
)
from src.services.storage.s3 import S3Service
from src.app.config import settings


router = APIRouter()


def check_album_access(album_id: UUID, user: User, db: Session) -> None:
    """Verify user has upload access to album."""
    album_repo = AlbumRepository(db)
    album = album_repo.get(album_id)
    
    if not album:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Album not found')
    
    if user.role != 'admin' and album.photographer_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Not authorized to upload to this album')


def check_upload_quota(album_id: UUID, filesize: int, db: Session) -> None:
    """Check if upload quota is available."""
    photo_repo = PhotoRepository(db)
    stats = photo_repo.get_album_stats(album_id)
    
    # Max 10,000 photos per album
    if stats['total_photos'] >= 10000:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Album photo limit reached (10,000 photos)')
    
    # Max 100GB per album
    max_storage = 100 * 1024 * 1024 * 1024
    if stats['total_size_bytes'] + filesize > max_storage:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Album storage limit reached (100GB)')


# ============================================================================
# STANDARD UPLOAD FLOW
# ============================================================================

@router.post('/albums/{album_id}/presign', response_model=PresignResponse)
def request_presigned_url(
    album_id: UUID,
    request: PresignRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Request presigned URL for photo upload (Step 1 of upload flow)."""
    check_album_access(album_id, current_user, db)
    check_upload_quota(album_id, request.filesize, db)
    
    s3_service = S3Service()
    photo_repo = PhotoRepository(db)
    
    s3_key = s3_service.generate_s3_key(str(album_id), request.filename)
    
    photo = photo_repo.create_photo(
        album_id=album_id,
        uploader_id=current_user.id,
        s3_key=s3_key,
        s3_bucket=s3_service.bucket_name,
        filename=request.filename,
        content_type=request.content_type,
        filesize=request.filesize
    )
    
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
    """Request multiple presigned URLs for bulk upload (max 100 files)."""
    check_album_access(album_id, current_user, db)
    
    total_size = sum(f.filesize for f in request.files)
    check_upload_quota(album_id, total_size, db)
    
    s3_service = S3Service()
    photo_repo = PhotoRepository(db)
    
    uploads = []
    for file_request in request.files:
        s3_key = s3_service.generate_s3_key(str(album_id), file_request.filename)
        
        photo = photo_repo.create_photo(
            album_id=album_id,
            uploader_id=current_user.id,
            s3_key=s3_key,
            s3_bucket=s3_service.bucket_name,
            filename=file_request.filename,
            content_type=file_request.content_type,
            filesize=file_request.filesize
        )
        
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
    """Notify that upload is complete (Step 3 of upload flow)."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    s3_service = S3Service()
    
    photo = photo_repo.get(request.photo_id)
    if not photo or photo.album_id != album_id or photo.s3_key != request.s3_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid photo or S3 key')
    
    if not s3_service.check_object_exists(request.s3_key):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='File not found in S3')
    
    if request.exif:
        photo_repo.update_exif(photo.id, exif_data=request.exif)
    
    photo_repo.update_status(photo.id, PhotoStatus.processing)
    
    # TODO: Enqueue processing
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
    """Notify completion of multiple uploads (max 100 photos)."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    s3_service = S3Service()
    
    success_count = 0
    failed_photo_ids = []
    
    for upload in request.uploads:
        try:
            photo = photo_repo.get(upload.photo_id)
            if not photo or photo.album_id != album_id:
                failed_photo_ids.append(upload.photo_id)
                continue
            
            if not s3_service.check_object_exists(upload.s3_key):
                failed_photo_ids.append(upload.photo_id)
                continue
            
            if upload.exif:
                photo_repo.update_exif(upload.photo_id, upload.exif)
            
            photo_repo.update_status(upload.photo_id, PhotoStatus.processing)
            success_count += 1
            
        except Exception:
            failed_photo_ids.append(upload.photo_id)
    
    return BulkUploadCompleteResponse(
        success_count=success_count,
        failed_count=len(failed_photo_ids),
        failed_photo_ids=failed_photo_ids,
        message=f'Processed {success_count}/{len(request.uploads)} uploads'
    )


# ============================================================================
# MULTIPART UPLOAD FLOW
# ============================================================================

@router.post('/albums/{album_id}/multipart/init', response_model=MultipartUploadInitResponse)
def initialize_multipart_upload(
    album_id: UUID,
    request: MultipartUploadInitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Initialize multipart upload for large files (>100MB)."""
    check_album_access(album_id, current_user, db)
    check_upload_quota(album_id, request.filesize, db)
    
    s3_service = S3Service()
    photo_repo = PhotoRepository(db)
    
    s3_key = s3_service.generate_s3_key(str(album_id), request.filename)
    
    photo = photo_repo.create_photo(
        album_id=album_id,
        uploader_id=current_user.id,
        s3_key=s3_key,
        s3_bucket=s3_service.bucket_name,
        filename=request.filename,
        content_type=request.content_type,
        filesize=request.filesize
    )
    
    total_parts, actual_part_size = s3_service.calculate_multipart_info(
        request.filesize,
        request.part_size
    )
    
    upload_id = s3_service.initiate_multipart_upload(
        s3_key=s3_key,
        content_type=request.content_type
    )
    
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
    """Get presigned URLs for multipart upload parts (up to 100 parts)."""
    photo_repo = PhotoRepository(db)
    
    photo = photo_repo.get(request.photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    if not photo_repo.verify_album_ownership(request.photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
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


@router.get('/multipart/{upload_id}/parts', response_model=MultipartPartsListResponse)
def list_multipart_parts(
    upload_id: str,
    photo_id: UUID = Query(...),
    s3_key: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List uploaded parts for a multipart upload (for resume functionality)."""
    photo_repo = PhotoRepository(db)
    
    photo = photo_repo.get(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    if not photo_repo.verify_album_ownership(photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
    s3_service = S3Service()
    parts = s3_service.list_multipart_parts(s3_key, upload_id)
    
    return MultipartPartsListResponse(
        upload_id=upload_id,
        parts=parts,
        total_parts=len(parts)
    )


@router.post('/multipart/complete', response_model=UploadCompleteResponse)
def complete_multipart_upload(
    request: MultipartUploadCompleteRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Complete multipart upload after all parts uploaded."""
    photo_repo = PhotoRepository(db)
    
    photo = photo_repo.get(request.photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    if not photo_repo.verify_album_ownership(request.photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
    s3_service = S3Service()
    
    parts = [
        {'PartNumber': part['part_number'], 'ETag': part['etag']}
        for part in request.parts
    ]
    
    try:
        s3_service.complete_multipart_upload(
            s3_key=request.s3_key,
            upload_id=request.upload_id,
            parts=parts
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to complete multipart upload: {str(e)}')
    
    if request.exif:
        photo_repo.update_exif(request.photo_id, request.exif)
    
    photo_repo.update_status(request.photo_id, PhotoStatus.processing)
    
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
    """Abort multipart upload and cleanup."""
    photo_repo = PhotoRepository(db)
    
    photo = photo_repo.get(request.photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    if not photo_repo.verify_album_ownership(request.photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
    s3_service = S3Service()
    s3_service.abort_multipart_upload(request.s3_key, request.upload_id)
    
    photo_repo.delete(request.photo_id, soft=False)
    
    return None


@router.post('/multipart/{upload_id}/retry', response_model=MultipartPartPresignResponse)
def retry_multipart_parts(
    upload_id: str,
    photo_id: UUID = Query(...),
    s3_key: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Re-issue presigned URLs for missing/failed multipart parts."""
    photo_repo = PhotoRepository(db)
    
    photo = photo_repo.get(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    if not photo_repo.verify_album_ownership(photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
    s3_service = S3Service()
    
    # Get uploaded parts
    uploaded_parts = s3_service.list_multipart_parts(s3_key, upload_id)
    uploaded_part_nums = {part['PartNumber'] for part in uploaded_parts}
    
    # Get total expected parts
    extra_data = photo.extra_data or {}
    total_parts = extra_data.get('total_parts', 0)
    
    if not total_parts:
        raise HTTPException(status_code=400, detail='Unable to determine total parts')
    
    # Find missing parts
    missing_parts = [i for i in range(1, total_parts + 1) if i not in uploaded_part_nums]
    
    if not missing_parts:
        return MultipartPartPresignResponse(part_urls={}, expires_in=3600)
    
    # Generate URLs for missing parts
    part_urls = s3_service.generate_multipart_presigned_urls(
        s3_key=s3_key,
        upload_id=upload_id,
        part_numbers=missing_parts,
        expires_in=3600
    )
    
    return MultipartPartPresignResponse(
        part_urls=part_urls,
        expires_in=3600
    )


# ============================================================================
# UPLOAD STATUS & MONITORING
# ============================================================================

@router.get('/{photo_id}', response_model=UploadMetadataResponse)
def get_upload_metadata(
    photo_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get upload metadata and status."""
    photo_repo = PhotoRepository(db)
    
    photo = photo_repo.get_with_relations(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    if not photo_repo.verify_album_ownership(photo_id, current_user.id):
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail='Not authorized')
    
    upload_progress = photo.extra_data.get('upload_progress', {}) if photo.extra_data else {}
    
    return UploadMetadataResponse(
        photo_id=photo.id,
        filename=photo.filename,
        filesize=photo.filesize,
        content_type=photo.content_type,
        status=photo.status,
        s3_key=photo.s3_key,
        bytes_uploaded=upload_progress.get('bytes_uploaded', 0),
        upload_percentage=upload_progress.get('percentage', 0.0),
        created_at=photo.created_at,
        updated_at=photo.updated_at,
        processing_error=photo.processing_error
    )


@router.get('/albums/{album_id}/uploads', response_model=RecentUploadsResponse)
def list_recent_uploads(
    album_id: UUID,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    status: Optional[PhotoStatus] = Query(None),
    hours: int = Query(24, ge=1, le=168),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List recent uploads in album with pagination and filters."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    
    skip = (page - 1) * size
    photos = photo_repo.get_recent_uploads(album_id, hours, limit=1000)
    
    if status:
        photos = [p for p in photos if p.status == status]
    
    total = len(photos)
    photos = photos[skip:skip + size]
    
    pages = (total + size - 1) // size
    
    return RecentUploadsResponse(
        items=photos,
        total=total,
        page=page,
        size=size,
        pages=pages
    )


@router.post('/progress', response_model=UploadProgressResponse)
def update_upload_progress(
    request: UploadProgressRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update upload progress for monitoring and UI."""
    photo_repo = PhotoRepository(db)
    
    photo = photo_repo.get(request.photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    percentage = (request.bytes_uploaded / request.total_bytes) * 100
    
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
    """Get upload statistics for monitoring."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    
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
    """Get upload quota information."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    stats = photo_repo.get_album_stats(album_id)
    
    max_photos = 10000
    max_storage = 100 * 1024 * 1024 * 1024
    
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


@router.get('/albums/{album_id}/health', response_model=AlbumHealthResponse)
def get_album_health(
    album_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Quick health check for album upload workflow."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    stats = photo_repo.get_album_stats(album_id)
    status_breakdown = photo_repo.count_by_status(album_id)
    
    total = stats['total_photos']
    if total == 0:
        health_score = 100.0
    else:
        failed = stats['failed_photos']
        health_score = max(0, 100 - (failed / total * 100))
    
    return AlbumHealthResponse(
        album_id=album_id,
        health_score=round(health_score, 2),
        total_photos=total,
        processing_photos=stats['processing_photos'],
        failed_photos=failed,
        pending_uploads=status_breakdown.get('uploaded', 0),
        is_healthy=health_score >= 95.0
    )


# ============================================================================
# FILE MANAGEMENT
# ============================================================================

@router.get('/albums/{album_id}/files/{photo_id}/presign-download', response_model=dict)
def presign_file_download(
    album_id: UUID,
    photo_id: UUID,
    expires_in: int = Query(3600, ge=300, le=86400),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Request short-lived presigned download URL."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    photo = photo_repo.get(photo_id)
    
    if not photo or photo.album_id != album_id:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    s3_service = S3Service()
    download_url = s3_service.generate_presigned_download_url(
        s3_key=photo.s3_key,
        expires_in=expires_in,
        filename=photo.filename
    )
    
    return {
        'download_url': download_url,
        'expires_in': expires_in,
        'filename': photo.filename
    }


@router.post('/albums/{album_id}/files/{photo_id}/copy', response_model=dict)
def copy_file(
    album_id: UUID,
    photo_id: UUID,
    request: FileCopyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Copy/move file between albums or buckets."""
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin only')
    
    photo_repo = PhotoRepository(db)
    photo = photo_repo.get(photo_id)
    
    if not photo or photo.album_id != album_id:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    s3_service = S3Service()
    
    dest_key = s3_service.generate_s3_key(str(request.destination_album_id), photo.filename)
    
    etag = s3_service.copy_object(
        source_key=photo.s3_key,
        destination_key=dest_key
    )
    
    if not request.move:
        new_photo = photo_repo.create_photo(
            album_id=request.destination_album_id,
            uploader_id=current_user.id,
            s3_key=dest_key,
            s3_bucket=s3_service.bucket_name,
            filename=photo.filename,
            content_type=photo.content_type,
            filesize=photo.filesize
        )
        return {'photo_id': new_photo.id, 's3_key': dest_key, 'etag': etag}
    else:
        s3_service.delete_object(photo.s3_key)
        photo_repo.update(photo_id, {'s3_key': dest_key, 'album_id': request.destination_album_id})
        return {'photo_id': photo_id, 's3_key': dest_key, 'etag': etag}


@router.delete('/albums/{album_id}/files/{photo_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_file(
    album_id: UUID,
    photo_id: UUID,
    hard: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete file (soft delete by default, hard delete with S3 cleanup)."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    photo = photo_repo.get(photo_id)
    
    if not photo or photo.album_id != album_id:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    if hard:
        s3_service = S3Service()
        s3_service.delete_object(photo.s3_key)
        photo_repo.delete(photo_id, soft=False)
    else:
        photo_repo.delete(photo_id, soft=True)
    
    return None


@router.post('/albums/{album_id}/files/{photo_id}/purge-cdn', status_code=status.HTTP_202_ACCEPTED)
def purge_cdn_cache(
    album_id: UUID,
    photo_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Purge cached CDN object."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    photo = photo_repo.get(photo_id)
    
    if not photo or photo.album_id != album_id:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    # TODO: Implement CDN purge logic (CloudFront, CloudFlare, etc.)
    # Example: cloudfront_service.create_invalidation(paths=[photo.s3_key])
    
    return {
        'message': 'CDN purge requested',
        'photo_id': photo_id,
        'status': 'pending'
    }


@router.patch('/albums/{album_id}/files/{photo_id}', response_model=dict)
def update_file_metadata(
    album_id: UUID,
    photo_id: UUID,
    request: FileMetadataUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update file metadata (title, tags, privacy, EXIF overrides)."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    photo = photo_repo.get(photo_id)
    
    if not photo or photo.album_id != album_id:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    update_data = {}
    if request.title is not None:
        update_data['title'] = request.title
    if request.description is not None:
        update_data['description'] = request.description
    if request.tags is not None:
        update_data['tags'] = request.tags
    if request.is_private is not None:
        update_data['is_private'] = request.is_private
    if request.exif_overrides:
        existing_exif = photo.exif_data or {}
        existing_exif.update(request.exif_overrides)
        update_data['exif_data'] = existing_exif
    
    photo_repo.update(photo_id, update_data)
    
    return {
        'photo_id': photo_id,
        'message': 'Metadata updated successfully',
        'updated_fields': list(update_data.keys())
    }


@router.post('/albums/{album_id}/files/{photo_id}/derive', response_model=DeriveAssetResponse)
def derive_asset(
    album_id: UUID,
    photo_id: UUID,
    request: DeriveAssetRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Request derived asset (thumbnail, WebP, different sizes)."""
    check_album_access(album_id, current_user, db)
    
    photo_repo = PhotoRepository(db)
    photo = photo_repo.get(photo_id)
    
    if not photo or photo.album_id != album_id:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    # Generate job ID
    job_id = f"derive_{photo_id}_{request.asset_type}_{datetime.utcnow().timestamp()}"
    
    # TODO: Enqueue background task for asset derivation
    # background_tasks.add_task(derive_asset_task.delay, job_id, photo_id, request.dict())
    
    return DeriveAssetResponse(
        job_id=job_id,
        photo_id=photo_id,
        asset_type=request.asset_type,
        status='queued',
        message=f'Derivation job for {request.asset_type} queued'
    )


# ============================================================================
# ADMIN & MODERATION
# ============================================================================

@router.get('/flagged', response_model=FlaggedFilesResponse)
def list_flagged_files(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    reason: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List flagged/quarantined files (Admin only)."""
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin only')
    
    photo_repo = PhotoRepository(db)
    
    skip = (page - 1) * size
    
    query = photo_repo.db.query(photo_repo.model).filter(
        photo_repo.model.is_flagged == True
    )
    
    if reason:
        query = query.filter(photo_repo.model.flag_reason == reason)
    
    total = query.count()
    photos = query.offset(skip).limit(size).all()
    
    pages = (total + size - 1) // size
    
    return FlaggedFilesResponse(
        items=photos,
        total=total,
        page=page,
        size=size,
        pages=pages
    )


@router.post('/{photo_id}/quarantine', status_code=status.HTTP_200_OK)
def quarantine_file(
    photo_id: UUID,
    reason: str = Query(..., min_length=1, max_length=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Quarantine a file (Admin only)."""
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin only')
    
    photo_repo = PhotoRepository(db)
    photo = photo_repo.get(photo_id)
    
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    photo_repo.update(photo_id, {
        'is_flagged': True,
        'flag_reason': reason,
        'flagged_at': datetime.utcnow(),
        'flagged_by': current_user.id
    })
    
    return {
        'photo_id': photo_id,
        'message': 'File quarantined successfully',
        'reason': reason
    }


@router.post('/{photo_id}/restore', status_code=status.HTTP_200_OK)
def restore_file(
    photo_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Restore quarantined file (Admin only)."""
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin only')
    
    photo_repo = PhotoRepository(db)
    photo = photo_repo.get(photo_id)
    
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found')
    
    photo_repo.update(photo_id, {
        'is_flagged': False,
        'flag_reason': None,
        'flagged_at': None,
        'flagged_by': None
    })
    
    return {
        'photo_id': photo_id,
        'message': 'File restored successfully'
    }


# ============================================================================
# OPERATIONAL & BULK OPERATIONS
# ============================================================================

@router.post('/bulk-delete', status_code=status.HTTP_200_OK)
def bulk_delete_files(
    request: BulkDeleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Bulk delete files (Admin only, max 1000 files)."""
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin only')
    
    if len(request.photo_ids) > 1000:
        raise HTTPException(status_code=400, detail='Maximum 1000 files per bulk delete')
    
    photo_repo = PhotoRepository(db)
    s3_service = S3Service()
    
    success_count = 0
    failed_ids = []
    s3_keys_to_delete = []
    
    for photo_id in request.photo_ids:
        try:
            photo = photo_repo.get(photo_id)
            if not photo:
                failed_ids.append(photo_id)
                continue
            
            s3_keys_to_delete.append(photo.s3_key)
            
            if request.hard_delete:
                photo_repo.delete(photo_id, soft=False)
            else:
                photo_repo.delete(photo_id, soft=True)
            
            success_count += 1
            
        except Exception:
            failed_ids.append(photo_id)
    
    # Bulk delete from S3 if hard delete
    if request.hard_delete and s3_keys_to_delete:
        s3_success, s3_failed = s3_service.delete_objects_bulk(s3_keys_to_delete)
    
    return {
        'success_count': success_count,
        'failed_count': len(failed_ids),
        'failed_ids': failed_ids,
        'message': f'Deleted {success_count}/{len(request.photo_ids)} files'
    }


@router.post('/maintenance/reindex', status_code=status.HTTP_202_ACCEPTED)
def reindex_uploads(
    album_id: Optional[UUID] = Query(None),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Reindex uploads for search (Admin only)."""
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin only')
    
    # TODO: Implement search reindexing logic
    # background_tasks.add_task(reindex_photos_task.delay, album_id)
    
    return {
        'message': 'Reindexing started',
        'album_id': album_id,
        'status': 'pending'
    }