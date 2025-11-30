"""Photo repository for database operations."""
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, and_, or_, desc, asc
from typing import Optional, List, Tuple
from uuid import UUID
from datetime import datetime, timedelta

from src.repositories.base import BaseRepository
from src.models.photo import Photo
from src.models.face import Face
from src.models.album import Album
from src.models.user import User
from src.models.enums import PhotoStatus


class PhotoRepository(BaseRepository[Photo]):
    """Repository for photo database operations."""
    
    def __init__(self, db: Session):
        super().__init__(Photo, db)
    
    def create_photo(
        self, 
        album_id: UUID, 
        uploader_id: UUID,
        s3_key: str,
        s3_bucket: str,
        filename: str,
        content_type: str,
        filesize: int
    ) -> Photo:
        """
        Create new photo record.
        
        Args:
            album_id: Album UUID
            uploader_id: Uploader UUID
            s3_key: S3 object key
            s3_bucket: S3 bucket name
            filename: Original filename
            content_type: MIME type
            filesize: File size in bytes
            
        Returns:
            Created Photo instance
        """
        photo_dict = {
            'album_id': album_id,
            'uploader_id': uploader_id,
            's3_key': s3_key,
            's3_bucket': s3_bucket,
            'filename': filename,
            'content_type': content_type,
            'filesize': filesize,
            'status': PhotoStatus.uploaded
        }
        return self.create(photo_dict)
    
    def get_by_s3_key(self, s3_key: str, include_deleted: bool = False) -> Optional[Photo]:
        """
        Get photo by S3 key.
        
        Args:
            s3_key: S3 object key
            include_deleted: Include soft-deleted photos
            
        Returns:
            Photo instance or None
        """
        return self.get_by_field('s3_key', s3_key, include_deleted)
    
    def get_with_relations(self, photo_id: UUID) -> Optional[Photo]:
        """
        Get photo with album, uploader, and faces eagerly loaded.
        
        Args:
            photo_id: Photo UUID
            
        Returns:
            Photo with relationships loaded
        """
        return self.db.query(Photo).options(
            joinedload(Photo.album),
            joinedload(Photo.uploader),
            joinedload(Photo.faces)
        ).filter(
            Photo.id == photo_id,
            Photo.deleted_at.is_(None)
        ).first()
    
    def get_by_album(
        self,
        album_id: UUID,
        skip: int = 0,
        limit: int = 100,
        status: Optional[PhotoStatus] = None,
        include_deleted: bool = False,
        order_by: str = 'created_at',
        order_desc: bool = True
    ) -> Tuple[List[Photo], int]:
        """
        Get all photos in an album with pagination and filtering.
        
        Args:
            album_id: Album UUID
            skip: Number of records to skip
            limit: Maximum records to return
            status: Filter by photo status
            include_deleted: Include soft-deleted photos
            order_by: Field to order by (created_at, filename, filesize, taken_at)
            order_desc: Order descending
            
        Returns:
            Tuple of (photos list, total count)
        """
        query = self.db.query(Photo).filter(Photo.album_id == album_id)
        
        if not include_deleted:
            query = query.filter(Photo.deleted_at.is_(None))
        
        if status:
            query = query.filter(Photo.status == status)
        
        total = query.count()
        
        # Ordering
        order_column = getattr(Photo, order_by, Photo.created_at)
        if order_desc:
            query = query.order_by(desc(order_column))
        else:
            query = query.order_by(asc(order_column))
        
        photos = query.offset(skip).limit(limit).all()
        return photos, total
    
    def get_with_face_count(self, photo_id: UUID) -> Optional[dict]:
        """
        Get photo with face count.
        
        Args:
            photo_id: Photo UUID
            
        Returns:
            Dict with photo and face_count, or None
        """
        photo = self.get(photo_id)
        if not photo:
            return None
        
        face_count = self.db.query(func.count(Face.id)).filter(
            Face.photo_id == photo_id,
            Face.deleted_at.is_(None)
        ).scalar() or 0
        
        return {
            'photo': photo,
            'face_count': face_count
        }
    
    def get_album_stats(self, album_id: UUID) -> dict:
        """
        Get comprehensive photo statistics for an album.
        
        Args:
            album_id: Album UUID
            
        Returns:
            Dict with various statistics
        """
        query = self.db.query(Photo).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None)
        )
        
        total_photos = query.count()
        
        # Count by status
        processed = query.filter(Photo.status == PhotoStatus.done).count()
        processing = query.filter(Photo.status == PhotoStatus.processing).count()
        failed = query.filter(Photo.status == PhotoStatus.failed).count()
        
        # Total storage size
        total_size = self.db.query(
            func.coalesce(func.sum(Photo.filesize), 0)
        ).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None)
        ).scalar() or 0
        
        # Count faces
        total_faces = self.db.query(func.count(Face.id)).join(Photo).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None),
            Face.deleted_at.is_(None)
        ).scalar() or 0
        
        # Count photos with at least one face
        photos_with_faces = self.db.query(
            func.count(func.distinct(Face.photo_id))
        ).join(Photo).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None),
            Face.deleted_at.is_(None)
        ).scalar() or 0
        
        return {
            'album_id': album_id,
            'total_photos': total_photos,
            'processed_photos': processed,
            'processing_photos': processing,
            'failed_photos': failed,
            'total_size_bytes': int(total_size),
            'total_faces': total_faces,
            'photos_with_faces': photos_with_faces
        }
    
    def update_status(
        self, 
        photo_id: UUID, 
        status: PhotoStatus,
        error: Optional[str] = None
    ) -> Optional[Photo]:
        """
        Update photo processing status.
        
        Args:
            photo_id: Photo UUID
            status: New status
            error: Error message if failed
            
        Returns:
            Updated Photo or None
        """
        update_dict = {'status': status}
        if error:
            update_dict['processing_error'] = error
        return self.update(photo_id, update_dict)
    
    def update_dimensions(
        self, 
        photo_id: UUID, 
        width: int, 
        height: int
    ) -> Optional[Photo]:
        """
        Update photo dimensions.
        
        Args:
            photo_id: Photo UUID
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Updated Photo or None
        """
        return self.update(photo_id, {'width': width, 'height': height})
    
    def update_thumbnails(
        self,
        photo_id: UUID,
        small_url: Optional[str] = None,
        medium_url: Optional[str] = None,
        large_url: Optional[str] = None,
        watermarked_url: Optional[str] = None
    ) -> Optional[Photo]:
        """
        Update photo thumbnail URLs.
        
        Args:
            photo_id: Photo UUID
            small_url: Small thumbnail URL
            medium_url: Medium thumbnail URL
            large_url: Large thumbnail URL
            watermarked_url: Watermarked version URL
            
        Returns:
            Updated Photo or None
        """
        update_dict = {}
        if small_url:
            update_dict['thumbnail_small_url'] = small_url
        if medium_url:
            update_dict['thumbnail_medium_url'] = medium_url
        if large_url:
            update_dict['thumbnail_large_url'] = large_url
        if watermarked_url:
            update_dict['watermarked_url'] = watermarked_url
        
        return self.update(photo_id, update_dict) if update_dict else None
    
    def update_exif(
        self,
        photo_id: UUID,
        exif_data: dict,
        taken_at: Optional[datetime] = None,
        camera_model: Optional[str] = None
    ) -> Optional[Photo]:
        """
        Update photo EXIF data and extracted metadata.
        
        Args:
            photo_id: Photo UUID
            exif_data: EXIF data dictionary
            taken_at: Photo taken timestamp
            camera_model: Camera model string
            
        Returns:
            Updated Photo or None
        """
        update_dict = {'exif': exif_data}
        if taken_at:
            update_dict['taken_at'] = taken_at
        if camera_model:
            update_dict['camera_model'] = camera_model
        
        return self.update(photo_id, update_dict)
    
    def get_unprocessed(self, limit: int = 100) -> List[Photo]:
        """
        Get photos that need processing.
        
        Args:
            limit: Maximum photos to return
            
        Returns:
            List of unprocessed photos
        """
        return self.db.query(Photo).filter(
            Photo.status == PhotoStatus.uploaded,
            Photo.deleted_at.is_(None)
        ).order_by(Photo.created_at).limit(limit).all()
    
    def get_failed_photos(
        self, 
        album_id: Optional[UUID] = None,
        skip: int = 0,
        limit: int = 100
    ) -> Tuple[List[Photo], int]:
        """
        Get failed photos with optional album filter.
        
        Args:
            album_id: Optional album UUID filter
            skip: Records to skip
            limit: Maximum records
            
        Returns:
            Tuple of (photos list, total count)
        """
        query = self.db.query(Photo).filter(
            Photo.status == PhotoStatus.failed,
            Photo.deleted_at.is_(None)
        )
        
        if album_id:
            query = query.filter(Photo.album_id == album_id)
        
        total = query.count()
        photos = query.order_by(desc(Photo.updated_at)).offset(skip).limit(limit).all()
        
        return photos, total
    
    def bulk_delete_by_album(
        self, 
        album_id: UUID,
        photo_ids: List[UUID]
    ) -> Tuple[int, List[UUID]]:
        """
        Bulk soft delete photos in an album.
        
        Args:
            album_id: Album UUID (for ownership verification)
            photo_ids: List of photo UUIDs to delete
            
        Returns:
            Tuple of (success count, failed IDs list)
        """
        photos = self.db.query(Photo).filter(
            Photo.id.in_(photo_ids),
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None)
        ).all()
        
        now = datetime.utcnow()
        success_count = 0
        for photo in photos:
            photo.deleted_at = now
            success_count += 1
        
        found_ids = {photo.id for photo in photos}
        failed_ids = [pid for pid in photo_ids if pid not in found_ids]
        
        self.commit()
        return success_count, failed_ids
    
    def bulk_update_status(
        self,
        photo_ids: List[UUID],
        status: PhotoStatus,
        album_id: Optional[UUID] = None
    ) -> Tuple[int, List[UUID]]:
        """
        Bulk update photo status.
        
        Args:
            photo_ids: List of photo UUIDs
            status: New status
            album_id: Optional album filter
            
        Returns:
            Tuple of (success count, failed IDs)
        """
        query = self.db.query(Photo).filter(
            Photo.id.in_(photo_ids),
            Photo.deleted_at.is_(None)
        )
        
        if album_id:
            query = query.filter(Photo.album_id == album_id)
        
        photos = query.all()
        
        success_count = 0
        for photo in photos:
            photo.status = status
            success_count += 1
        
        found_ids = {photo.id for photo in photos}
        failed_ids = [pid for pid in photo_ids if pid not in found_ids]
        
        self.commit()
        return success_count, failed_ids
    
    def reprocess_photo(self, photo_id: UUID) -> Optional[Photo]:
        """
        Mark photo for reprocessing.
        
        Args:
            photo_id: Photo UUID
            
        Returns:
            Updated Photo or None
        """
        return self.update_status(photo_id, PhotoStatus.uploaded)
    
    def get_recent_uploads(
        self,
        album_id: UUID,
        hours: int = 24,
        limit: int = 50
    ) -> List[Photo]:
        """
        Get recent photo uploads within specified hours.
        
        Args:
            album_id: Album UUID
            hours: Hours to look back
            limit: Maximum photos to return
            
        Returns:
            List of recent photos
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return self.db.query(Photo).filter(
            Photo.album_id == album_id,
            Photo.created_at >= cutoff,
            Photo.deleted_at.is_(None)
        ).order_by(desc(Photo.created_at)).limit(limit).all()
    
    def search_photos(
        self,
        album_id: UUID,
        search_term: str,
        skip: int = 0,
        limit: int = 100
    ) -> Tuple[List[Photo], int]:
        """
        Search photos by filename or camera model.
        
        Args:
            album_id: Album UUID
            search_term: Search query
            skip: Records to skip
            limit: Maximum records
            
        Returns:
            Tuple of (photos list, total count)
        """
        query = self.db.query(Photo).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None)
        )
        
        search_pattern = f'%{search_term}%'
        search_filter = or_(
            Photo.filename.ilike(search_pattern),
            Photo.camera_model.ilike(search_pattern)
        )
        query = query.filter(search_filter)
        
        total = query.count()
        photos = query.order_by(desc(Photo.created_at)).offset(skip).limit(limit).all()
        
        return photos, total
    
    def count_by_status(self, album_id: UUID) -> dict:
        """
        Count photos by status in an album.
        
        Args:
            album_id: Album UUID
            
        Returns:
            Dict mapping status to count
        """
        query = self.db.query(
            Photo.status,
            func.count(Photo.id).label('count')
        ).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None)
        ).group_by(Photo.status)
        
        # Initialize with all statuses
        results = {status.value: 0 for status in PhotoStatus}
        
        # Fill in actual counts
        for status, count in query.all():
            results[status.value] = count
        
        return results
    
    def get_photos_by_uploader(
        self,
        uploader_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> Tuple[List[Photo], int]:
        """
        Get all photos uploaded by a specific user.
        
        Args:
            uploader_id: Uploader UUID
            skip: Records to skip
            limit: Maximum records
            
        Returns:
            Tuple of (photos list, total count)
        """
        return self.get_multi_by_field(
            'uploader_id',
            uploader_id,
            skip=skip,
            limit=limit
        )
    
    def get_photos_with_faces(
        self,
        album_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> Tuple[List[Photo], int]:
        """
        Get photos that have detected faces.
        
        Args:
            album_id: Album UUID
            skip: Records to skip
            limit: Maximum records
            
        Returns:
            Tuple of (photos list, total count)
        """
        # Subquery to get photo IDs with faces
        photos_with_faces_subquery = self.db.query(Face.photo_id).filter(
            Face.deleted_at.is_(None)
        ).distinct().subquery()
        
        query = self.db.query(Photo).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None),
            Photo.id.in_(photos_with_faces_subquery)
        )
        
        total = query.count()
        photos = query.order_by(desc(Photo.created_at)).offset(skip).limit(limit).all()
        
        return photos, total
    
    def get_photos_without_faces(
        self,
        album_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> Tuple[List[Photo], int]:
        """
        Get photos without detected faces.
        
        Args:
            album_id: Album UUID
            skip: Records to skip
            limit: Maximum records
            
        Returns:
            Tuple of (photos list, total count)
        """
        # Subquery to get photo IDs with faces
        photos_with_faces_subquery = self.db.query(Face.photo_id).filter(
            Face.deleted_at.is_(None)
        ).distinct().subquery()
        
        query = self.db.query(Photo).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None),
            Photo.status == PhotoStatus.done,  # Only processed photos
            ~Photo.id.in_(photos_with_faces_subquery)
        )
        
        total = query.count()
        photos = query.order_by(desc(Photo.created_at)).offset(skip).limit(limit).all()
        
        return photos, total
    
    def verify_album_ownership(self, photo_id: UUID, user_id: UUID) -> bool:
        """
        Verify user owns the album containing this photo.
        
        Args:
            photo_id: Photo UUID
            user_id: User UUID
            
        Returns:
            True if user owns the album, False otherwise
        """
        photo = self.get_with_relations(photo_id)
        if not photo or not photo.album:
            return False
        return photo.album.photographer_id == user_id
    
    def get_by_date_range(
        self,
        album_id: UUID,
        start_date: datetime,
        end_date: datetime,
        skip: int = 0,
        limit: int = 100
    ) -> Tuple[List[Photo], int]:
        """
        Get photos within a date range (by taken_at or created_at).
        
        Args:
            album_id: Album UUID
            start_date: Range start
            end_date: Range end
            skip: Records to skip
            limit: Maximum records
            
        Returns:
            Tuple of (photos list, total count)
        """
        query = self.db.query(Photo).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None),
            or_(
                and_(Photo.taken_at >= start_date, Photo.taken_at <= end_date),
                and_(
                    Photo.taken_at.is_(None),
                    Photo.created_at >= start_date,
                    Photo.created_at <= end_date
                )
            )
        )
        
        total = query.count()
        photos = query.order_by(
            desc(func.coalesce(Photo.taken_at, Photo.created_at))
        ).offset(skip).limit(limit).all()
        
        return photos, total
    
    def get_large_files(
        self,
        album_id: UUID,
        min_size_bytes: int,
        skip: int = 0,
        limit: int = 100
    ) -> Tuple[List[Photo], int]:
        """
        Get photos larger than specified size.
        
        Args:
            album_id: Album UUID
            min_size_bytes: Minimum file size
            skip: Records to skip
            limit: Maximum records
            
        Returns:
            Tuple of (photos list, total count)
        """
        query = self.db.query(Photo).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None),
            Photo.filesize >= min_size_bytes
        )
        
        total = query.count()
        photos = query.order_by(desc(Photo.filesize)).offset(skip).limit(limit).all()
        
        return photos, total
    
    def get_processing_queue_size(self) -> int:
        """
        Get count of photos waiting to be processed.
        
        Returns:
            Count of photos in uploaded status
        """
        return self.count(
            include_deleted=False,
            filters={'status': PhotoStatus.uploaded}
        )
    
    def get_total_storage_by_album(self, album_id: UUID) -> int:
        """
        Get total storage used by album in bytes.
        
        Args:
            album_id: Album UUID
            
        Returns:
            Total bytes used
        """
        total = self.db.query(
            func.coalesce(func.sum(Photo.filesize), 0)
        ).filter(
            Photo.album_id == album_id,
            Photo.deleted_at.is_(None)
        ).scalar()
        
        return int(total or 0)