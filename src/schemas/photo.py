"""Photo schemas for API requests and responses."""
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime
from uuid import UUID
from typing import Optional, Any
from src.models.enums import PhotoStatus


class PhotoUpdate(BaseModel):
    """Schema for updating photo metadata."""
    filename: Optional[str] = Field(None, min_length=1, max_length=255)
    taken_at: Optional[datetime] = None
    camera_model: Optional[str] = Field(None, max_length=255)
    extra_data: Optional[dict[str, Any]] = None


class PhotoResponse(BaseModel):
    """Photo response with URLs."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    album_id: UUID
    uploader_id: Optional[UUID] = None
    filename: str
    content_type: str
    filesize: int
    width: Optional[int] = None
    height: Optional[int] = None
    status: PhotoStatus
    processing_error: Optional[str] = None
    taken_at: Optional[datetime] = None
    camera_model: Optional[str] = None
    
    # URLs
    thumbnail_small_url: Optional[str] = None
    thumbnail_medium_url: Optional[str] = None
    thumbnail_large_url: Optional[str] = None
    watermarked_url: Optional[str] = None
    original_url: Optional[str] = None  # Generated presigned URL
    
    # Metadata
    exif: Optional[dict[str, Any]] = None
    extra_data: Optional[dict[str, Any]] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    # Face count
    face_count: int = 0


class PhotoDetailResponse(PhotoResponse):
    """Extended photo response with uploader and album details."""
    uploader_name: Optional[str] = None
    uploader_email: Optional[str] = None
    album_title: Optional[str] = None


class PhotoListResponse(BaseModel):
    """Paginated list of photos."""
    items: list[PhotoResponse]
    total: int
    page: int
    size: int
    pages: int


class PhotoStatsResponse(BaseModel):
    """Photo statistics for an album."""
    album_id: UUID
    total_photos: int
    processed_photos: int
    processing_photos: int
    failed_photos: int
    total_size_bytes: int
    total_faces: int
    photos_with_faces: int


class PhotoBulkActionRequest(BaseModel):
    """Request for bulk photo actions."""
    photo_ids: list[UUID] = Field(..., min_items=1, max_items=100, description="Max 100 photos")
    action: str = Field(..., pattern="^(delete|reprocess|download)$", description="Action to perform")


class PhotoBulkActionResponse(BaseModel):
    """Response for bulk photo actions."""
    success_count: int
    failed_count: int
    failed_ids: list[UUID]
    message: str
    download_url: Optional[str] = None  # For bulk download


class PhotoDownloadRequest(BaseModel):
    """Request to download photo."""
    quality: str = Field(
        default='original', 
        pattern="^(thumbnail|medium|high|original)$",
        description="Quality/size of download"
    )
    watermark: bool = Field(
        default=True, 
        description="Include watermark (if available)"
    )


class PhotoDownloadResponse(BaseModel):
    """Response with download URL."""
    download_url: str
    expires_in: int = 3600  # seconds
    filename: str


class PhotoSearchRequest(BaseModel):
    """Request to search photos."""
    query: str = Field(..., min_length=1, max_length=255)
    search_in: list[str] = Field(
        default=['filename', 'camera_model'],
        description="Fields to search in"
    )


class PhotoFilterRequest(BaseModel):
    """Advanced photo filtering."""
    status: Optional[list[PhotoStatus]] = None
    has_faces: Optional[bool] = None
    min_width: Optional[int] = Field(None, gt=0)
    max_width: Optional[int] = Field(None, gt=0)
    min_height: Optional[int] = Field(None, gt=0)
    max_height: Optional[int] = Field(None, gt=0)
    min_filesize: Optional[int] = Field(None, gt=0)
    max_filesize: Optional[int] = Field(None, gt=0)
    taken_after: Optional[datetime] = None
    taken_before: Optional[datetime] = None
    camera_models: Optional[list[str]] = None