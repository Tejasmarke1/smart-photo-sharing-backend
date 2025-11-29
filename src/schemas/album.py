"""Album schemas."""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from uuid import UUID
from typing import Optional, Any


class AlbumBase(BaseModel):
    """Base album schema with common fields."""
    title: str = Field(..., min_length=1, max_length=255, description="Album title")
    description: Optional[str] = Field(None, description="Album description")
    location: Optional[str] = Field(None, max_length=255, description="Event location")
    start_time: Optional[datetime] = Field(None, description="Event start time")
    end_time: Optional[datetime] = Field(None, description="Event end time")
    
    # Sharing configuration
    is_public: bool = Field(False, description="Whether album is publicly accessible")
    consent_required: bool = Field(True, description="Whether guest consent is required")
    password_protected: bool = Field(False, description="Whether album requires password")
    album_password: Optional[str] = Field(None, description="Album password (will be hashed)")
    
    # Feature flags
    face_detection_enabled: bool = Field(True, description="Enable face detection")
    watermark_enabled: bool = Field(True, description="Enable watermarks on photos")
    download_enabled: bool = Field(True, description="Allow photo downloads")
    
    # Metadata
    cover_photo_url: Optional[str] = Field(None, max_length=512, description="Cover photo URL")
    extra_data: Optional[dict[str, Any]] = Field(None, description="Additional metadata")
    
    @field_validator('end_time')
    @classmethod
    def validate_end_time(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Ensure end_time is after start_time."""
        if v and info.data.get('start_time') and v < info.data['start_time']:
            raise ValueError('end_time must be after start_time')
        return v


class AlbumCreate(BaseModel):
    """Schema for creating an album."""
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    location: Optional[str] = Field(None, max_length=255)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    is_public: bool = False
    consent_required: bool = True
    password_protected: bool = False
    album_password: Optional[str] = None
    
    face_detection_enabled: bool = True
    watermark_enabled: bool = True
    download_enabled: bool = True
    
    cover_photo_url: Optional[str] = Field(None, max_length=512)
    extra_data: Optional[dict[str, Any]] = None
    
    @field_validator('end_time')
    @classmethod
    def validate_end_time(cls, v: Optional[datetime], info) -> Optional[datetime]:
        if v and info.data.get('start_time') and v < info.data['start_time']:
            raise ValueError('end_time must be after start_time')
        return v
    
    @field_validator('album_password')
    @classmethod
    def validate_password(cls, v: Optional[str], info) -> Optional[str]:
        """Validate password when password protection is enabled."""
        if info.data.get('password_protected') and not v:
            raise ValueError('album_password is required when password_protected is True')
        if v and len(v) < 6:
            raise ValueError('album_password must be at least 6 characters')
        return v


class AlbumUpdate(BaseModel):
    """Schema for updating an album."""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    location: Optional[str] = Field(None, max_length=255)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    is_public: Optional[bool] = None
    consent_required: Optional[bool] = None
    password_protected: Optional[bool] = None
    album_password: Optional[str] = None
    
    face_detection_enabled: Optional[bool] = None
    watermark_enabled: Optional[bool] = None
    download_enabled: Optional[bool] = None
    
    cover_photo_url: Optional[str] = Field(None, max_length=512)
    extra_data: Optional[dict[str, Any]] = None


class AlbumInDB(AlbumBase):
    """Schema for album as stored in database."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    photographer_id: UUID
    sharing_code: str
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None


class AlbumResponse(BaseModel):
    """Schema for album response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    photographer_id: UUID
    title: str
    description: Optional[str] = None
    location: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    sharing_code: str
    is_public: bool
    consent_required: bool
    password_protected: bool
    
    face_detection_enabled: bool
    watermark_enabled: bool
    download_enabled: bool
    
    cover_photo_url: Optional[str] = None
    extra_data: Optional[dict[str, Any]] = None
    
    created_at: datetime
    updated_at: datetime
    
    # Computed fields
    photo_count: int = 0
    face_count: int = 0
    person_count: int = 0


class AlbumDetailResponse(AlbumResponse):
    """Extended album response with photographer details."""
    photographer_name: Optional[str] = None
    photographer_email: Optional[str] = None


class AlbumListResponse(BaseModel):
    """Paginated list of albums."""
    items: list[AlbumResponse]
    total: int
    page: int
    size: int
    pages: int


class AlbumShareResponse(BaseModel):
    """Response for album sharing information."""
    sharing_code: str
    share_url: str
    qr_code_url: str
    is_public: bool
    password_protected: bool


class AlbumPasswordVerify(BaseModel):
    """Schema for verifying album password."""
    password: str = Field(..., min_length=1, description="Album password to verify")


class AlbumStatsResponse(BaseModel):
    """Album statistics response."""
    album_id: UUID
    photo_count: int
    face_count: int
    person_count: int
    total_size_bytes: int
    upload_count_today: int
    last_upload_at: Optional[datetime] = None


class AlbumBulkActionRequest(BaseModel):
    """Request for bulk album actions."""
    album_ids: list[UUID] = Field(..., min_items=1, max_items=100)
    action: str = Field(..., pattern="^(delete|archive|activate)$")


class AlbumBulkActionResponse(BaseModel):
    """Response for bulk album actions."""
    success_count: int
    failed_count: int
    failed_ids: list[UUID]
    message: str