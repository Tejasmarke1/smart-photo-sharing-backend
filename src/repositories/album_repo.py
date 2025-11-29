"""Album repository extending base repository."""
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, and_, or_, desc
from typing import Optional, List, Tuple
from uuid import UUID
from datetime import datetime, timedelta
import secrets

from src.repositories.base import BaseRepository
from src.models.album import Album
from src.models.photo import Photo
from src.models.face import Face
from src.models.person import Person
from src.schemas.album import AlbumCreate, AlbumUpdate
from src.core.security import hash_password


class AlbumRepository(BaseRepository[Album]):
    """Repository for album database operations."""
    
    def __init__(self, db: Session):
        super().__init__(Album, db)
    
    @staticmethod
    def generate_sharing_code(length: int = 16) -> str:
        """Generate URL-safe sharing code."""
        return secrets.token_urlsafe(length)
    
    def create_album(self, album_data: AlbumCreate, photographer_id: UUID) -> Album:
        """
        Create new album with unique sharing code.
        
        Args:
            album_data: Album creation data
            photographer_id: UUID of the photographer creating the album
            
        Returns:
            Created Album instance
        """
        # Generate unique sharing code
        sharing_code = self.generate_sharing_code()
        while self.get_by_sharing_code(sharing_code):
            sharing_code = self.generate_sharing_code()
        
        # Hash password if provided
        hashed_password = None
        if album_data.password_protected and album_data.album_password:
            hashed_password = hash_password(album_data.album_password)
        
        # Prepare album data
        album_dict = album_data.model_dump(exclude={'album_password'})
        album_dict['photographer_id'] = photographer_id
        album_dict['sharing_code'] = sharing_code
        album_dict['album_password'] = hashed_password
        
        # Use base create method
        return self.create(album_dict)
    
    def get_by_sharing_code(self, sharing_code: str, include_deleted: bool = False) -> Optional[Album]:
        """
        Get album by sharing code.
        
        Args:
            sharing_code: Unique sharing code
            include_deleted: Whether to include soft-deleted albums
            
        Returns:
            Album instance or None if not found
        """
        return self.get_by_field('sharing_code', sharing_code, include_deleted)
    
    def get_with_photographer(self, album_id: UUID) -> Optional[Album]:
        """
        Get album with photographer details eagerly loaded.
        
        Args:
            album_id: Album UUID
            
        Returns:
            Album instance with photographer relationship loaded
        """
        return self.db.query(Album).options(
            joinedload(Album.photographer)
        ).filter(
            Album.id == album_id,
            Album.deleted_at.is_(None)
        ).first()
    
    def get_all_by_photographer(
        self,
        photographer_id: UUID,
        skip: int = 0,
        limit: int = 100,
        include_deleted: bool = False,
        search: Optional[str] = None,
        order_by: str = 'created_at',
        order_desc: bool = True
    ) -> Tuple[List[Album], int]:
        """
        Get all albums by photographer with optional search and pagination.
        
        Args:
            photographer_id: Photographer UUID
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return
            include_deleted: Whether to include soft-deleted albums
            search: Search term for title, description, or location
            order_by: Field to order by (created_at, title, start_time)
            order_desc: Whether to order descending
            
        Returns:
            Tuple of (list of albums, total count)
        """
        if search:
            # Build query with photographer filter and search
            query = self.db.query(Album).filter(
                Album.photographer_id == photographer_id
            )
            
            if not include_deleted:
                query = query.filter(Album.deleted_at.is_(None))
            
            search_term = f'%{search}%'
            search_filter = or_(
                Album.title.ilike(search_term),
                Album.description.ilike(search_term),
                Album.location.ilike(search_term)
            )
            query = query.filter(search_filter)
            
            total = query.count()
            
            # Apply ordering
            order_column = getattr(Album, order_by, Album.created_at)
            if order_desc:
                query = query.order_by(desc(order_column))
            else:
                query = query.order_by(order_column)
            
            albums = query.offset(skip).limit(limit).all()
            return albums, total
        
        # Use base get_multi with photographer filter
        return self.get_multi(
            skip=skip,
            limit=limit,
            include_deleted=include_deleted,
            order_by=order_by,
            order_desc=order_desc,
            filters={'photographer_id': photographer_id}
        )
    
    def get_with_counts(self, album_id: UUID) -> Optional[dict]:
        """
        Get album with photo, face, and person counts.
        
        Args:
            album_id: Album UUID
            
        Returns:
            Dict with album and counts, or None if album not found
        """
        album = self.get(album_id)
        if not album:
            return None
        
        # Count active photos
        photo_count = self.db.query(func.count(Photo.id)).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None)
        ).scalar() or 0
        
        # Count faces from active photos
        face_count = self.db.query(func.count(Face.id)).join(Photo).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None),
            Face.deleted_at.is_(None)
        ).scalar() or 0
        
        # Count persons
        person_count = self.db.query(func.count(Person.id)).filter(
            Person.album_id == album_id,
            Person.deleted_at.is_(None)
        ).scalar() or 0
        
        return {
            'album': album,
            'photo_count': photo_count,
            'face_count': face_count,
            'person_count': person_count
        }
    
    def get_stats(self, album_id: UUID) -> Optional[dict]:
        """
        Get detailed statistics for an album.
        
        Args:
            album_id: Album UUID
            
        Returns:
            Dict with detailed statistics
        """
        album = self.get(album_id)
        if not album:
            return None
        
        # Get basic counts
        counts = self.get_with_counts(album_id)
        
        # Get total storage size
        total_size = self.db.query(func.coalesce(func.sum(Photo.filesize), 0)).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None)
        ).scalar() or 0
        
        # Get today's upload count
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        upload_count_today = self.db.query(func.count(Photo.id)).filter(
            Photo.album_id == album_id,
            Photo.created_at >= today_start,
            Photo.deleted_at.is_(None)
        ).scalar() or 0
        
        # Get last upload time
        last_photo = self.db.query(Photo).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None)
        ).order_by(desc(Photo.created_at)).first()
        
        return {
            'album_id': album_id,
            'photo_count': counts['photo_count'],
            'face_count': counts['face_count'],
            'person_count': counts['person_count'],
            'total_size_bytes': int(total_size),
            'upload_count_today': upload_count_today,
            'last_upload_at': last_photo.created_at if last_photo else None
        }
    
    def update_album(self, album_id: UUID, album_data: AlbumUpdate) -> Optional[Album]:
        """
        Update album.
        
        Args:
            album_id: Album UUID
            album_data: Update data
            
        Returns:
            Updated Album instance or None if not found
        """
        # Get update dict excluding unset fields
        update_dict = album_data.model_dump(exclude_unset=True, exclude={'album_password'})
        
        # Handle password update separately with hashing
        if album_data.album_password is not None:
            update_dict['album_password'] = hash_password(album_data.album_password)
        
        # Use base update method
        return self.update(album_id, update_dict)
    
    def regenerate_sharing_code(self, album_id: UUID) -> Optional[str]:
        """
        Regenerate sharing code for album (invalidates old links).
        
        Args:
            album_id: Album UUID
            
        Returns:
            New sharing code or None if album not found
        """
        album = self.get(album_id)
        if not album:
            return None
        
        # Generate new unique code
        new_code = self.generate_sharing_code()
        while self.get_by_sharing_code(new_code):
            new_code = self.generate_sharing_code()
        
        # Update using base method
        updated = self.update(album_id, {'sharing_code': new_code})
        return updated.sharing_code if updated else None
    
    def update_cover_photo(self, album_id: UUID, photo_url: str) -> bool:
        """
        Update album cover photo URL.
        
        Args:
            album_id: Album UUID
            photo_url: URL of the photo to set as cover
            
        Returns:
            True if successful, False if album not found
        """
        updated = self.update(album_id, {'cover_photo_url': photo_url})
        return updated is not None
    
    def bulk_soft_delete_by_photographer(
        self, 
        album_ids: List[UUID], 
        photographer_id: UUID
    ) -> Tuple[int, List[UUID]]:
        """
        Bulk soft delete albums belonging to photographer.
        
        Args:
            album_ids: List of album UUIDs to delete
            photographer_id: Photographer UUID (for ownership verification)
            
        Returns:
            Tuple of (success count, list of failed album IDs)
        """
        # Get albums that exist and belong to photographer
        albums = self.db.query(Album).filter(
            Album.id.in_(album_ids),
            Album.photographer_id == photographer_id,
            Album.deleted_at.is_(None)
        ).all()
        
        # Mark as deleted
        now = datetime.utcnow()
        success_count = 0
        for album in albums:
            album.deleted_at = now
            success_count += 1
        
        # Identify failed IDs (albums not found or not owned by photographer)
        found_ids = {album.id for album in albums}
        failed_ids = [aid for aid in album_ids if aid not in found_ids]
        
        self.commit()
        return success_count, failed_ids
    
    def bulk_restore_by_photographer(
        self, 
        album_ids: List[UUID], 
        photographer_id: UUID
    ) -> Tuple[int, List[UUID]]:
        """
        Bulk restore soft-deleted albums.
        
        Args:
            album_ids: List of album UUIDs to restore
            photographer_id: Photographer UUID (for ownership verification)
            
        Returns:
            Tuple of (success count, list of failed album IDs)
        """
        # Get soft-deleted albums that belong to photographer
        albums = self.db.query(Album).filter(
            Album.id.in_(album_ids),
            Album.photographer_id == photographer_id,
            Album.deleted_at.isnot(None)
        ).all()
        
        # Restore albums
        success_count = 0
        for album in albums:
            album.deleted_at = None
            success_count += 1
        
        # Identify failed IDs
        found_ids = {album.id for album in albums}
        failed_ids = [aid for aid in album_ids if aid not in found_ids]
        
        self.commit()
        return success_count, failed_ids
    
    def get_albums_by_date_range(
        self,
        photographer_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> List[Album]:
        """
        Get albums within a date range based on event start/end times.
        
        Args:
            photographer_id: Photographer UUID
            start_date: Range start date
            end_date: Range end date
            
        Returns:
            List of albums within the date range
        """
        return self.db.query(Album).filter(
            Album.photographer_id == photographer_id,
            Album.deleted_at.is_(None),
            or_(
                # Album starts within range
                and_(Album.start_time >= start_date, Album.start_time <= end_date),
                # Album ends within range
                and_(Album.end_time >= start_date, Album.end_time <= end_date),
                # Album spans the entire range
                and_(Album.start_time <= start_date, Album.end_time >= end_date)
            )
        ).order_by(Album.start_time).all()
    
    def get_public_albums(
        self, 
        skip: int = 0, 
        limit: int = 50,
        search: Optional[str] = None
    ) -> Tuple[List[Album], int]:
        """
        Get public albums with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            search: Optional search term
            
        Returns:
            Tuple of (list of albums, total count)
        """
        filters = {'is_public': True}
        
        if search:
            # Use search with public filter
            query = self.db.query(Album).filter(
                Album.is_public == True,
                Album.deleted_at.is_(None)
            )
            
            search_term = f'%{search}%'
            search_filter = or_(
                Album.title.ilike(search_term),
                Album.description.ilike(search_term),
                Album.location.ilike(search_term)
            )
            query = query.filter(search_filter)
            
            total = query.count()
            albums = query.order_by(desc(Album.created_at)).offset(skip).limit(limit).all()
            return albums, total
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            order_by='created_at',
            order_desc=True,
            filters=filters
        )
    
    def get_recent_albums(
        self, 
        photographer_id: UUID, 
        days: int = 30, 
        limit: int = 10
    ) -> List[Album]:
        """
        Get recent albums created within specified days.
        
        Args:
            photographer_id: Photographer UUID
            days: Number of days to look back
            limit: Maximum number of albums to return
            
        Returns:
            List of recent albums
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        return self.db.query(Album).filter(
            Album.photographer_id == photographer_id,
            Album.created_at >= cutoff_date,
            Album.deleted_at.is_(None)
        ).order_by(desc(Album.created_at)).limit(limit).all()
    
    def count_by_photographer(self, photographer_id: UUID) -> dict:
        """
        Get album counts by status for a photographer.
        
        Args:
            photographer_id: Photographer UUID
            
        Returns:
            Dict with active, deleted, and total counts
        """
        total = self.count(include_deleted=True, filters={'photographer_id': photographer_id})
        active = self.count(include_deleted=False, filters={'photographer_id': photographer_id})
        deleted = total - active
        
        return {
            'total': total,
            'active': active,
            'deleted': deleted
        }
    
    def is_owner(self, album_id: UUID, photographer_id: UUID) -> bool:
        """
        Check if photographer owns the album.
        
        Args:
            album_id: Album UUID
            photographer_id: Photographer UUID
            
        Returns:
            True if photographer owns the album, False otherwise
        """
        album = self.get(album_id)
        return album is not None and album.photographer_id == photographer_id