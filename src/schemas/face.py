from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic import ValidationInfo




class FaceBase(BaseModel):
    """Base face schema."""
    bbox: Dict[str, int] = Field(..., description="Bounding box {x, y, w, h}")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    
    @field_validator('bbox')
    def validate_bbox(cls, v: Dict[str, int]):
        required = {'x', 'y', 'w', 'h'}
        if not all(k in v for k in required):
            raise ValueError(f"bbox must contain {required}")
        if any(v[k] < 0 for k in required):
            raise ValueError("bbox values must be non-negative")
        return v


class FaceCreate(FaceBase):
    """Schema for creating a face."""
    photo_id: UUID
    thumbnail_s3_key: Optional[str] = None
    blur_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    brightness_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class FaceResponse(FaceBase):
    """Schema for face response."""
    id: UUID
    photo_id: UUID
    thumbnail_url: Optional[str] = None
    blur_score: Optional[float] = None
    brightness_score: Optional[float] = None
    person_id: Optional[UUID] = None
    person_name: Optional[str] = None
    created_at: datetime

    # Pydantic v2 config
    model_config = ConfigDict(from_attributes=True)


class FaceLabelRequest(BaseModel):
    """Schema for labeling a face."""
    person_id: Optional[UUID] = Field(None, description="Existing person ID")
    person_name: Optional[str] = Field(None, description="New person name")
    is_manual: bool = Field(True, description="Manual vs automatic assignment")
    
    @field_validator('person_name')
    def validate_person_name(cls, v: Optional[str], info: ValidationInfo):
        data = info.data or {}
        if not data.get('person_id') and not v:
            raise ValueError("Either person_id or person_name must be provided")
        if v and len(v.strip()) < 2:
            raise ValueError("Person name must be at least 2 characters")
        return v.strip() if v else v


class FaceSearchRequest(BaseModel):
    """Schema for face search by embedding."""
    embedding: List[float] = Field(..., min_items=512, max_items=512)
    album_id: Optional[UUID] = None
    k: int = Field(50, ge=1, le=200, description="Number of results")
    threshold: float = Field(0.6, ge=0.0, le=1.0, description="Similarity threshold")
    min_quality: Optional[float] = Field(None, ge=0.0, le=1.0)


class FaceSearchResponse(BaseModel):
    """Schema for search results."""
    face_id: UUID
    photo_id: UUID
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    thumbnail_url: Optional[str]
    person_id: Optional[UUID] = None
    person_name: Optional[str] = None
    bbox: Dict[str, int]
    confidence: float


class FaceClusterRequest(BaseModel):
    """Schema for album clustering request."""
    min_cluster_size: int = Field(5, ge=2, description="Minimum faces per cluster")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    merge_threshold: float = Field(0.8, ge=0.0, le=1.0)


class FaceClusterResponse(BaseModel):
    """Schema for clustering results."""
    album_id: UUID
    total_faces: int
    num_clusters: int
    clusters: Dict[str, List[UUID]] = Field(..., description="cluster_id -> [face_ids]")
    merge_suggestions: List[Dict[str, Any]]


class FaceQualityFilter(BaseModel):
    """Schema for quality filtering."""
    min_blur_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_brightness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    
class FaceListResponse(BaseModel):
    """Schema for listing faces with pagination."""
    faces: List[FaceResponse]
    total: int
    page: int
    size: int


class FaceListResult(BaseModel):
    """Schema for listing faces with skip/limit pagination."""
    faces: List[FaceResponse]
    total: int
    skip: int
    limit: int
    
    model_config = ConfigDict(from_attributes=True)


class JobAccepted(BaseModel):
    """Schema for async job acceptance responses (202)."""
    message: str
    job_id: str
    status_url: str
    photo_id: Optional[str] = None
    album_id: Optional[str] = None