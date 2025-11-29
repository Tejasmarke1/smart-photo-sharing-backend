"""Album API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
import qrcode
from io import BytesIO
import base64

from src.api.deps import get_db, get_current_user
from src.models.user import User
from src.repositories.album_repo import AlbumRepository
from src.schemas.album import (
    AlbumCreate,
    AlbumUpdate,
    AlbumResponse,
    AlbumDetailResponse,
    AlbumListResponse,
    AlbumShareResponse,
    AlbumPasswordVerify,
    AlbumStatsResponse,
    AlbumBulkActionRequest,
    AlbumBulkActionResponse
)
from src.core.security import verify_password
from src.app.config import settings


router = APIRouter()


def build_album_response(album_data: dict) -> AlbumResponse:
    """Build AlbumResponse from album and counts."""
    album = album_data['album']
    return AlbumResponse(
        **album.__dict__,
        photo_count=album_data.get('photo_count', 0),
        face_count=album_data.get('face_count', 0),
        person_count=album_data.get('person_count', 0)
    )


@router.post('/', response_model=AlbumResponse, status_code=status.HTTP_201_CREATED)
def create_album(
    album_data: AlbumCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create new album (authenticated users).
    
    - **title**: Album title (required)
    - **location**: Event location
    - **start_time**: Event start time
    - **end_time**: Event end time
    - **is_public**: Make album publicly accessible
    - **password_protected**: Require password for access
    - **album_password**: Password (if password_protected is True)
    - **face_detection_enabled**: Enable face detection
    - **watermark_enabled**: Add watermarks to photos
    - **download_enabled**: Allow photo downloads
    """
    repo = AlbumRepository(db)
    album = repo.create_album(album_data, current_user.id)
    
    return AlbumResponse(
        **album.__dict__,
        photo_count=0,
        face_count=0,
        person_count=0
    )


@router.get('/{album_id}', response_model=AlbumDetailResponse)
def get_album(
    album_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get album details with counts.
    
    Users can only access their own albums unless they are admin.
    """
    repo = AlbumRepository(db)
    
    # Get album with photographer details
    album = repo.get_with_photographer(album_id)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    # Check ownership (users can only see their own albums, unless admin)
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized to access this album'
        )
    
    # Get counts
    album_data = repo.get_with_counts(album_id)
    
    return AlbumDetailResponse(
        **album.__dict__,
        photo_count=album_data['photo_count'],
        face_count=album_data['face_count'],
        person_count=album_data['person_count'],
        photographer_name=album.photographer.name if album.photographer else None,
        photographer_email=album.photographer.email if album.photographer else None
    )


@router.get('/', response_model=AlbumListResponse)
def list_albums(
    page: int = Query(1, ge=1, description='Page number'),
    size: int = Query(20, ge=1, le=100, description='Page size'),
    search: Optional[str] = Query(None, description='Search in title, description, location'),
    include_deleted: bool = Query(False, description='Include soft-deleted albums'),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all albums for current user with pagination and search.
    
    - **page**: Page number (default: 1)
    - **size**: Items per page (default: 20, max: 100)
    - **search**: Search term for title, description, or location
    - **include_deleted**: Include soft-deleted albums
    
    Admins can see all albums, regular users see only their own.
    """
    repo = AlbumRepository(db)
    skip = (page - 1) * size
    
    # Admins can see all albums, others see only their own
    photographer_id = None if current_user.role == 'admin' else current_user.id
    
    if photographer_id:
        albums, total = repo.get_all_by_photographer(
            photographer_id=photographer_id,
            skip=skip,
            limit=size,
            include_deleted=include_deleted,
            search=search
        )
    else:
        # Admin: get all albums
        albums, total = repo.get_multi(
            skip=skip,
            limit=size,
            include_deleted=include_deleted,
            order_by='created_at',
            order_desc=True
        )
    
    # Build responses with counts
    album_responses = []
    for album in albums:
        album_data = repo.get_with_counts(album.id)
        album_responses.append(build_album_response(album_data))
    
    pages = (total + size - 1) // size
    
    return AlbumListResponse(
        items=album_responses,
        total=total,
        page=page,
        size=size,
        pages=pages
    )


@router.patch('/{album_id}', response_model=AlbumResponse)
def update_album(
    album_id: UUID,
    album_data: AlbumUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update album details.
    
    Only album owner or admin can update an album.
    """
    repo = AlbumRepository(db)
    
    # Verify ownership
    album = repo.get(album_id)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    # Check authorization
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized to update this album'
        )
    
    # Update album
    updated_album = repo.update_album(album_id, album_data)
    if not updated_album:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Failed to update album'
        )
    
    # Get counts
    album_with_counts = repo.get_with_counts(album_id)
    return build_album_response(album_with_counts)


@router.delete('/{album_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_album(
    album_id: UUID,
    hard: bool = Query(False, description='Permanent deletion'),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete album (soft delete by default, hard delete with query param).
    
    - **hard=false**: Soft delete (recoverable via restore endpoint)
    - **hard=true**: Permanent deletion (triggers background cleanup of photos and S3 objects)
    
    Only album owner or admin can delete an album.
    """
    repo = AlbumRepository(db)
    
    # Verify ownership
    album = repo.get(album_id, include_deleted=True)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    # Check authorization
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized to delete this album'
        )
    
    # Delete album
    if hard:
        # TODO: Enqueue background job for S3 cleanup
        # from src.tasks.workers.cleanup import enqueue_album_deletion
        # enqueue_album_deletion(album_id)
        success = repo.delete(album_id, soft=False)
    else:
        success = repo.delete(album_id, soft=True)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Failed to delete album'
        )
    
    return None


@router.post('/{album_id}/restore', response_model=AlbumResponse)
def restore_album(
    album_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Restore soft-deleted album.
    
    Only album owner or admin can restore an album.
    """
    repo = AlbumRepository(db)
    
    # Verify ownership
    album = repo.get(album_id, include_deleted=True)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    # Check authorization
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized to restore this album'
        )
    
    if not album.deleted_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Album is not deleted'
        )
    
    success = repo.restore(album_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Failed to restore album'
        )
    
    # Get restored album with counts
    album_data = repo.get_with_counts(album_id)
    return build_album_response(album_data)


@router.get('/{album_id}/share', response_model=AlbumShareResponse)
def get_share_link(
    album_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate shareable link and QR code for album.
    
    Returns share URL and base64-encoded QR code image.
    Only album owner or admin can generate share links.
    """
    repo = AlbumRepository(db)
    
    # Verify ownership
    album = repo.get(album_id)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    # Check authorization
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized to share this album'
        )
    
    # Build share URL
    share_url = f"{settings.FRONTEND_URL}/public/{album.sharing_code}"
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4
    )
    qr.add_data(share_url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color='black', back_color='white')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    qr_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return AlbumShareResponse(
        sharing_code=album.sharing_code,
        share_url=share_url,
        qr_code_url=f'data:image/png;base64,{qr_base64}',
        is_public=album.is_public,
        password_protected=album.password_protected
    )


@router.post('/{album_id}/regenerate-code', response_model=dict)
def regenerate_sharing_code(
    album_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Regenerate sharing code (invalidates old share links).
    
    Use this when you need to revoke access via old share links.
    Only album owner or admin can regenerate codes.
    """
    repo = AlbumRepository(db)
    
    # Verify ownership
    album = repo.get(album_id)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    # Check authorization
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized to regenerate code for this album'
        )
    
    new_code = repo.regenerate_sharing_code(album_id)
    if not new_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Failed to regenerate sharing code'
        )
    
    return {
        'sharing_code': new_code,
        'share_url': f"{settings.FRONTEND_URL}/public/{new_code}"
    }


@router.post('/{album_id}/verify-password', response_model=dict)
def verify_album_password(
    album_id: UUID,
    password_data: AlbumPasswordVerify,
    db: Session = Depends(get_db)
):
    """
    Verify album password for password-protected albums.
    
    Public endpoint - no authentication required.
    Returns 200 if password is valid, 401 if invalid.
    """
    repo = AlbumRepository(db)
    
    album = repo.get(album_id)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    if not album.password_protected:
        return {'valid': True, 'message': 'Album is not password protected'}
    
    if not album.album_password:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Album password not configured'
        )
    
    is_valid = verify_password(password_data.password, album.album_password)
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid password'
        )
    
    return {'valid': True, 'message': 'Password verified'}


@router.get('/{album_id}/stats', response_model=AlbumStatsResponse)
def get_album_stats(
    album_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed statistics for an album.
    
    Includes photo count, face count, storage usage, and upload activity.
    Only album owner or admin can view stats.
    """
    repo = AlbumRepository(db)
    
    # Verify ownership
    album = repo.get(album_id)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    # Check authorization
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized to view stats for this album'
        )
    
    stats = repo.get_stats(album_id)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Stats not available'
        )
    
    return AlbumStatsResponse(**stats)


@router.post('/bulk-action', response_model=AlbumBulkActionResponse)
def bulk_album_action(
    request: AlbumBulkActionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Perform bulk action on multiple albums.
    
    Supported actions:
    - **delete**: Soft delete albums
    - **archive**: Mark albums as inactive (soft delete)
    - **activate**: Restore soft-deleted albums
    
    Users can only perform actions on their own albums.
    Admins can perform actions on any albums.
    """
    repo = AlbumRepository(db)
    
    # Determine photographer_id based on role
    photographer_id = None if current_user.role == 'admin' else current_user.id
    
    if request.action == 'delete' or request.action == 'archive':
        if photographer_id:
            success_count, failed_ids = repo.bulk_soft_delete_by_photographer(
                request.album_ids, 
                photographer_id
            )
        else:
            # Admin can delete any album
            success_count, failed_ids = repo.bulk_delete(
                request.album_ids,
                soft=True
            )
        message = f"Successfully {request.action}d {success_count} album(s)"
    
    elif request.action == 'activate':
        if photographer_id:
            success_count, failed_ids = repo.bulk_restore_by_photographer(
                request.album_ids,
                photographer_id
            )
        else:
            # Admin can restore any album
            success_count = 0
            failed_ids = []
            for album_id in request.album_ids:
                if repo.restore(album_id):
                    success_count += 1
                else:
                    failed_ids.append(album_id)
        
        message = f"Successfully activated {success_count} album(s)"
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported action: {request.action}"
        )
    
    return AlbumBulkActionResponse(
        success_count=success_count,
        failed_count=len(failed_ids),
        failed_ids=failed_ids,
        message=message
    )


@router.patch('/{album_id}/cover', response_model=dict)
def update_album_cover(
    album_id: UUID,
    photo_url: str = Query(..., description='Photo URL to set as cover'),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update album cover photo.
    
    Only album owner or admin can update the cover photo.
    """
    repo = AlbumRepository(db)
    
    # Verify ownership
    album = repo.get(album_id)
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Album not found'
        )
    
    # Check authorization
    if current_user.role != 'admin' and album.photographer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Not authorized to update this album'
        )
    
    success = repo.update_cover_photo(album_id, photo_url)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Failed to update cover photo'
        )
    
    return {'message': 'Cover photo updated successfully', 'cover_photo_url': photo_url}