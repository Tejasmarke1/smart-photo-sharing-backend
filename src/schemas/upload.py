"""Upload schemas for photo upload flow."""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Any
from uuid import UUID
from datetime import datetime


class PresignRequest(BaseModel):
    """Request for presigned URL (single file)."""
    filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
    content_type: str = Field(default='image/jpeg', description="MIME type")
    filesize: int = Field(..., gt=0, le=100*1024*1024, description="File size in bytes (max 100MB)")
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Validate content type is an allowed image format."""
        allowed_types = [
            'image/jpeg', 'image/jpg', 'image/png', 
            'image/webp', 'image/heic', 'image/heif', 'image/tiff'
        ]
        if v.lower() not in allowed_types:
            raise ValueError(f'Content type must be one of: {", ".join(allowed_types)}')
        return v.lower()
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename has extension."""
        if '.' not in v:
            raise ValueError('Filename must have an extension')
        return v


class PresignResponse(BaseModel):
    """Response with presigned URL for upload."""
    upload_url: str = Field(..., description="Presigned URL for PUT/POST request")
    photo_id: UUID = Field(..., description="Photo record ID (for notification)")
    s3_key: str = Field(..., description="S3 object key")
    s3_bucket: str = Field(..., description="S3 bucket name")
    fields: Optional[dict[str, str]] = Field(None, description="Form fields for multipart upload")
    upload_id: Optional[str] = Field(None, description="Multipart upload ID (for large files)")
    expires_in: int = Field(default=3600, description="URL expiration in seconds")
    method: str = Field(default='PUT', description="HTTP method (PUT or POST)")


class BulkPresignRequest(BaseModel):
    """Request for multiple presigned URLs."""
    files: List[PresignRequest] = Field(..., min_items=1, max_items=100, description="Max 100 files")


class BulkPresignResponse(BaseModel):
    """Response with multiple presigned URLs."""
    uploads: List[PresignResponse]
    total: int
    estimated_total_size: int = Field(..., description="Total size in bytes")


class UploadCompleteRequest(BaseModel):
    """Notification that upload is complete."""
    photo_id: UUID = Field(..., description="Photo ID from presign response")
    s3_key: str = Field(..., description="S3 key from presign response")
    etag: Optional[str] = Field(None, description="S3 ETag from upload response")
    exif: Optional[dict[str, Any]] = Field(None, description="EXIF data (if extracted client-side)")
    
    @field_validator('s3_key')
    @classmethod
    def validate_s3_key(cls, v: str) -> str:
        """Validate S3 key format."""
        if not v.startswith('albums/'):
            raise ValueError('Invalid S3 key format')
        return v


class UploadCompleteResponse(BaseModel):
    """Response after upload completion."""
    model_config = ConfigDict(from_attributes=True)
    
    photo_id: UUID
    status: str
    message: str
    processing_started: bool = Field(..., description="Whether background processing started")


class BulkUploadCompleteRequest(BaseModel):
    """Notification for multiple completed uploads."""
    uploads: List[UploadCompleteRequest] = Field(..., min_items=1, max_items=100)


class BulkUploadCompleteResponse(BaseModel):
    """Response for bulk upload completion."""
    success_count: int
    failed_count: int
    failed_photo_ids: List[UUID]
    message: str


class MultipartUploadInitRequest(BaseModel):
    """Request to initialize multipart upload for large files."""
    filename: str = Field(..., min_length=1, max_length=255)
    content_type: str = Field(default='image/jpeg')
    filesize: int = Field(..., gt=5*1024*1024, description="Must be > 5MB for multipart")
    part_size: int = Field(default=5*1024*1024, ge=5*1024*1024, description="Part size (min 5MB)")
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/heic', 'image/heif', 'image/tiff']
        if v.lower() not in allowed_types:
            raise ValueError(f'Content type must be one of: {", ".join(allowed_types)}')
        return v.lower()


class MultipartUploadInitResponse(BaseModel):
    """Response with multipart upload details."""
    photo_id: UUID
    upload_id: str = Field(..., description="S3 multipart upload ID")
    s3_key: str
    s3_bucket: str
    total_parts: int = Field(..., description="Number of parts to upload")
    part_size: int = Field(..., description="Size of each part in bytes")


class MultipartPartPresignRequest(BaseModel):
    """Request presigned URLs for multipart parts."""
    photo_id: UUID
    upload_id: str
    s3_key: str
    part_numbers: List[int] = Field(..., min_items=1, max_items=100, description="Part numbers to get URLs for")
    
    @field_validator('part_numbers')
    @classmethod
    def validate_part_numbers(cls, v: List[int]) -> List[int]:
        """Validate part numbers are positive and sorted."""
        if any(p <= 0 for p in v):
            raise ValueError('Part numbers must be positive')
        return sorted(set(v))  # Remove duplicates and sort


class MultipartPartPresignResponse(BaseModel):
    """Response with presigned URLs for parts."""
    part_urls: dict[int, str] = Field(..., description="Map of part_number to presigned URL")
    expires_in: int = 3600


class MultipartUploadCompleteRequest(BaseModel):
    """Request to complete multipart upload."""
    photo_id: UUID
    upload_id: str
    s3_key: str
    parts: List[dict[str, Any]] = Field(..., description="List of {part_number, etag}")
    exif: Optional[dict[str, Any]] = None
    
    @field_validator('parts')
    @classmethod
    def validate_parts(cls, v: List[dict]) -> List[dict]:
        """Validate parts have required fields."""
        for part in v:
            if 'part_number' not in part or 'etag' not in part:
                raise ValueError('Each part must have part_number and etag')
            if not isinstance(part['part_number'], int) or part['part_number'] <= 0:
                raise ValueError('Invalid part_number')
        return sorted(v, key=lambda x: x['part_number'])


class MultipartUploadAbortRequest(BaseModel):
    """Request to abort multipart upload."""
    photo_id: UUID
    upload_id: str
    s3_key: str


class UploadProgressRequest(BaseModel):
    """Update upload progress (for monitoring)."""
    photo_id: UUID
    bytes_uploaded: int = Field(..., ge=0)
    total_bytes: int = Field(..., gt=0)
    
    @field_validator('bytes_uploaded')
    @classmethod
    def validate_progress(cls, v: int, info) -> int:
        """Validate uploaded bytes don't exceed total."""
        if 'total_bytes' in info.data and v > info.data['total_bytes']:
            raise ValueError('bytes_uploaded cannot exceed total_bytes')
        return v


class UploadProgressResponse(BaseModel):
    """Upload progress information."""
    photo_id: UUID
    bytes_uploaded: int
    total_bytes: int
    percentage: float = Field(..., ge=0, le=100)
    status: str


class UploadStatsResponse(BaseModel):
    """Upload statistics for monitoring."""
    album_id: UUID
    total_uploads_today: int
    successful_uploads_today: int
    failed_uploads_today: int
    total_bytes_uploaded_today: int
    average_upload_time_seconds: Optional[float] = None
    pending_uploads: int = Field(..., description="Photos in uploaded status")


class UploadQuotaResponse(BaseModel):
    """Upload quota information."""
    album_id: UUID
    max_photos: Optional[int] = Field(None, description="Max photos allowed (None = unlimited)")
    current_photos: int
    remaining_photos: Optional[int] = Field(None, description="Photos remaining (None = unlimited)")
    max_storage_bytes: Optional[int] = Field(None, description="Max storage (None = unlimited)")
    current_storage_bytes: int
    remaining_storage_bytes: Optional[int] = None
    can_upload: bool
    reason: Optional[str] = Field(None, description="Reason if can_upload is False")
    

class UploadMetadataResponse(BaseModel):
    """Response for upload metadata."""
    model_config = ConfigDict(from_attributes=True)
    
    photo_id: UUID
    filename: str
    filesize: int
    content_type: str
    status: str
    s3_key: str
    bytes_uploaded: int = 0
    upload_percentage: float = 0.0
    created_at: datetime
    updated_at: datetime
    processing_error: Optional[str] = None


class RecentUploadsResponse(BaseModel):
    """Response for paginated upload list."""
    items: List[Any]  # List of Photo models
    total: int
    page: int
    size: int
    pages: int


class FileMetadataUpdate(BaseModel):
    """Request to update file metadata."""
    title: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    tags: Optional[List[str]] = Field(None, max_items=50)
    is_private: Optional[bool] = None
    exif_overrides: Optional[dict[str, Any]] = Field(None, description="EXIF data to override")


class FileCopyRequest(BaseModel):
    """Request to copy/move file."""
    destination_album_id: UUID
    move: bool = Field(False, description="If True, move instead of copy")


class DeriveAssetRequest(BaseModel):
    """Request to derive asset (thumbnail, WebP, etc.)."""
    asset_type: str = Field(..., description="Type: thumbnail, webp, resize")
    size: Optional[str] = Field(None, description="For thumbnail: small, medium, large")
    width: Optional[int] = Field(None, ge=1, le=10000)
    height: Optional[int] = Field(None, ge=1, le=10000)
    quality: Optional[int] = Field(85, ge=1, le=100)
    format: Optional[str] = Field(None, description="Output format: jpeg, png, webp")
    
    @property
    def validate_asset_type(self):
        allowed = ['thumbnail', 'webp', 'resize', 'watermark']
        if self.asset_type not in allowed:
            raise ValueError(f'asset_type must be one of: {", ".join(allowed)}')
        return self.asset_type


class DeriveAssetResponse(BaseModel):
    """Response for asset derivation request."""
    job_id: str
    photo_id: UUID
    asset_type: str
    status: str = Field(..., description="Status: queued, processing, completed, failed")
    message: str
    estimated_completion: Optional[datetime] = None


class MultipartPartsListResponse(BaseModel):
    """Response listing uploaded multipart parts."""
    upload_id: str
    parts: List[dict[str, Any]] = Field(..., description="List of uploaded parts with PartNumber, Size, ETag")
    total_parts: int


class FlaggedFilesResponse(BaseModel):
    """Response for flagged files list."""
    items: List[Any]  # List of Photo models
    total: int
    page: int
    size: int
    pages: int


class BulkDeleteRequest(BaseModel):
    """Request for bulk file deletion."""
    photo_ids: List[UUID] = Field(..., min_items=1, max_items=1000)
    hard_delete: bool = Field(False, description="If True, permanently delete from S3")


class AlbumHealthResponse(BaseModel):
    """Response for album health check."""
    album_id: UUID
    health_score: float = Field(..., ge=0, le=100, description="Health score 0-100")
    total_photos: int
    processing_photos: int
    failed_photos: int
    pending_uploads: int
    is_healthy: bool = Field(..., description="True if health_score >= 95")