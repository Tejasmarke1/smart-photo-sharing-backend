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


class FaceInDB(BaseModel):
    """Schema for face with database fields."""
    id: UUID
    photo_id: UUID
    bbox: Dict[str, int]
    confidence: float
    thumbnail_s3_key: Optional[str] = None
    blur_score: Optional[float] = None
    brightness_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

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
    
class FaceQualityResponse(BaseModel):
    """Face quality assessment response."""
    face_id: UUID
    photo_id: UUID
    blur_score: Optional[float] = None
    brightness_score: Optional[float] = None
    confidence: float
    overall_quality: float
    quality_grade: str  # A, B, C, D, F
    issues: List[str] = []
    thumbnail_url: Optional[str] = None


class QualityCheckRequest(BaseModel):
    """Request for batch quality check."""
    face_ids: List[UUID] = Field(..., min_items=1, max_items=100)
    strict_mode: bool = Field(False, description="Use stricter quality thresholds")
    

class AdvancedFilterRequest(BaseModel):
    """Advanced face filtering criteria."""
    album_ids: Optional[List[UUID]] = None
    person_ids: Optional[List[UUID]] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_blur_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_brightness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    unlabeled_only: bool = False
    has_thumbnail: Optional[bool] = None
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    limit: int = Field(100, ge=1, le=1000)
    
class DuplicateFace(BaseModel):
    """Duplicate face pair."""
    face_id_1: UUID
    face_id_2: UUID
    similarity_score: float
    photo_id_1: UUID
    photo_id_2: UUID
    thumbnail_url_1: Optional[str] = None
    thumbnail_url_2: Optional[str] = None
    
class OutlierFace(BaseModel):
    """Face that doesn't match its person cluster."""
    face_id: UUID
    photo_id: UUID
    person_id: UUID
    person_name: str
    confidence: float
    avg_similarity_to_cluster: float
    thumbnail_url: Optional[str] = None
    
class ReprocessResponse(BaseModel):
    """Response for reprocessing operations."""
    message: str
    job_id: Optional[str] = None
    face_ids: Optional[List[UUID]] = None
    status_url: Optional[str] = None
    
    
class BatchReprocessRequest(BaseModel):
    """Request for batch reprocessing."""
    min_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Only reprocess faces below this quality")
    force_all: bool = Field(False, description="Reprocess all faces regardless of quality")
    
class BatchDeleteRequest(BaseModel):
    """Request for batch face deletion."""
    face_ids: List[UUID] = Field(..., min_items=1, max_items=500)
    delete_thumbnails: bool = Field(True, description="Also delete S3 thumbnails")
    
    
class AlbumDetectRequest(BaseModel):
    photo_ids: Optional[List[UUID]] = Field(None, description="Process only these photos; default is all in album")


# ============================================================================
# Advanced Clustering Schemas
# ============================================================================

class AutoClusterRequest(BaseModel):
    """Auto-cluster with smart defaults."""
    use_smart_defaults: bool = Field(True, description="Use smart algorithm-determined parameters")
    min_cluster_size: Optional[int] = Field(None, ge=2, le=100)
    similarity_threshold: Optional[float] = Field(None, ge=0.5, le=0.95)


class ClusterStatusResponse(BaseModel):
    """Clustering job status."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    total_faces: Optional[int] = None
    num_clusters: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class ClusterDetailResponse(BaseModel):
    """Detailed cluster information."""
    id: UUID
    album_id: UUID
    cluster_label: int
    size: int
    status: str
    avg_similarity: Optional[float]
    confidence_score: Optional[float]
    representative_face_id: Optional[UUID]
    representative_thumbnail_url: Optional[str]
    face_ids: List[UUID]
    sample_thumbnails: List[str]  # First 6 faces
    person_id: Optional[UUID] = None
    person_name: Optional[str] = None
    created_at: datetime


class ClusterResultsResponse(BaseModel):
    """Complete clustering results."""
    job_id: str
    album_id: UUID
    total_faces: int
    num_clusters: int
    clusters: List[ClusterDetailResponse]
    noise_faces: List[UUID]  # Faces not in any cluster
    merge_suggestions: List[Dict[str, Any]]
    completed_at: datetime


class ClusterReviewRequest(BaseModel):
    """Submit cluster review with corrections."""
    reviews: List[Dict[str, Any]] = Field(
        ...,
        description="List of {cluster_id, action, data} where action = accept|reject|split|merge"
    )
    
    @field_validator('reviews')
    def validate_reviews(cls, v: List[Dict[str, Any]]):
        valid_actions = {'accept', 'reject', 'split', 'merge'}
        for review in v:
            if 'cluster_id' not in review:
                raise ValueError("Each review must have cluster_id")
            if 'action' not in review or review['action'] not in valid_actions:
                raise ValueError(f"Each review must have action in {valid_actions}")
        return v


# ============================================================================
# Cluster Operations Schemas
# ============================================================================

class ClusterAcceptRequest(BaseModel):
    """Accept cluster as person."""
    person_name: str = Field(..., min_length=2, max_length=255)
    person_email: Optional[str] = None
    person_phone: Optional[str] = None


class ClusterSplitRequest(BaseModel):
    """Split cluster into multiple clusters."""
    face_groups: List[List[UUID]] = Field(
        ...,
        min_items=2,
        description="Split faces into N groups, each becoming a separate cluster"
    )
    
    @field_validator('face_groups')
    def validate_groups(cls, v: List[List[UUID]]):
        if len(v) < 2:
            raise ValueError("Must split into at least 2 groups")
        for group in v:
            if len(group) < 1:
                raise ValueError("Each group must have at least 1 face")
        return v


class ClusterMergeRequest(BaseModel):
    """Merge multiple clusters."""
    cluster_ids: List[UUID] = Field(..., min_items=2, max_items=10)
    person_name: Optional[str] = Field(None, min_length=2, max_length=255)


# ============================================================================
# Auto-labeling Schemas
# ============================================================================

class AutoLabelRequest(BaseModel):
    """Auto-label faces based on existing persons."""
    album_id: UUID
    person_ids: Optional[List[UUID]] = Field(None, description="Limit to specific persons")
    min_confidence: float = Field(0.8, ge=0.6, le=0.99, description="Minimum similarity for auto-label")
    unlabeled_only: bool = Field(True, description="Only label unlabeled faces")
    max_faces: int = Field(100, ge=1, le=1000, description="Max faces to label")


class LabelSuggestion(BaseModel):
    """Auto-labeling suggestion."""
    face_id: UUID
    photo_id: UUID
    thumbnail_url: Optional[str]
    person_id: UUID
    person_name: str
    similarity_score: float
    confidence: str  # high, medium, low
    reasoning: str


class LabelSuggestionsResponse(BaseModel):
    """Collection of label suggestions."""
    album_id: UUID
    suggestions: List[LabelSuggestion]
    total: int


class ConfirmLabelRequest(BaseModel):
    """Confirm auto-label."""
    person_id: UUID
    confidence_override: Optional[float] = Field(None, ge=0.0, le=1.0)


class AutoLabelResponse(BaseModel):
    """Auto-labeling operation result."""
    message: str
    labeled_count: int
    skipped_count: int
    face_ids: List[UUID]
    person_assignments: Dict[str, UUID]  # face_id -> person_id