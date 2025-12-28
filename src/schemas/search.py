"""Search-related schemas for face recognition."""
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Multi-face Search Schemas
# ============================================================================

class MultiFaceSearchRequest(BaseModel):
    """Search using multiple face embeddings."""
    face_ids: Optional[List[UUID]] = Field(None, description="Face IDs to search with")
    selfie_images: Optional[List[str]] = Field(None, description="Base64 encoded images")
    album_ids: Optional[List[UUID]] = Field(None, description="Limit to specific albums")
    k: int = Field(50, ge=1, le=500, description="Results per face")
    threshold: float = Field(0.6, ge=0.0, le=1.0)
    aggregation: str = Field("union", description="union, intersection, weighted")
    
    @field_validator('aggregation')
    def validate_aggregation(cls, v: str):
        if v not in ['union', 'intersection', 'weighted']:
            raise ValueError("aggregation must be union, intersection, or weighted")
        return v


class PersonSearchRequest(BaseModel):
    """Search all photos containing a person."""
    person_id: UUID
    album_ids: Optional[List[UUID]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_face_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    include_metadata: bool = Field(True, description="Include photo metadata")
    limit: int = Field(100, ge=1, le=1000)


class CrossAlbumSearchRequest(BaseModel):
    """Search across multiple albums."""
    face_id: Optional[UUID] = None
    selfie_image: Optional[str] = Field(None, description="Base64 encoded image")
    album_ids: List[UUID] = Field(..., min_items=1, max_items=50)
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    group_by_album: bool = Field(True, description="Group results by album")
    limit_per_album: int = Field(20, ge=1, le=100)


# ============================================================================
# Similarity Search Variation Schemas
# ============================================================================

class ThresholdScanRequest(BaseModel):
    """Find optimal similarity threshold."""
    face_id: Optional[UUID] = None
    selfie_image: Optional[str] = None
    album_id: UUID
    target_result_count: int = Field(50, ge=10, le=200, description="Desired number of results")
    min_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_threshold: float = Field(0.95, ge=0.0, le=1.0)
    step: float = Field(0.05, ge=0.01, le=0.1)


class ThresholdScanResult(BaseModel):
    """Threshold scan results."""
    threshold: float
    result_count: int
    avg_similarity: float
    sample_results: List[Dict[str, Any]]


class ThresholdScanResponse(BaseModel):
    """Complete threshold scan response."""
    recommended_threshold: float
    scan_results: List[ThresholdScanResult]
    reasoning: str


class ContextualSearchRequest(BaseModel):
    """Search with metadata context."""
    face_id: Optional[UUID] = None
    selfie_image: Optional[str] = None
    album_id: UUID
    
    # Context filters
    date_range: Optional[Dict[str, str]] = Field(None, description="from/to dates")
    location_proximity: Optional[Dict[str, Any]] = Field(None, description="lat/lng/radius")
    event_tags: Optional[List[str]] = None
    photo_quality_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Weights
    similarity_weight: float = Field(0.7, ge=0.0, le=1.0)
    context_weight: float = Field(0.3, ge=0.0, le=1.0)
    
    threshold: float = Field(0.6, ge=0.0, le=1.0)
    limit: int = Field(50, ge=1, le=200)


class ProgressiveSearchRequest(BaseModel):
    """Progressive search configuration."""
    face_id: Optional[UUID] = None
    selfie_image: Optional[str] = None
    album_id: UUID
    
    # Progressive stages
    initial_threshold: float = Field(0.85, ge=0.5, le=0.99, description="Start with high precision")
    final_threshold: float = Field(0.6, ge=0.3, le=0.95, description="End with high recall")
    stages: int = Field(3, ge=2, le=5, description="Number of refinement stages")
    
    max_results: int = Field(100, ge=10, le=500)
    timeout_seconds: int = Field(30, ge=5, le=120)


class ProgressiveSearchStage(BaseModel):
    """Single stage in progressive search."""
    stage: int
    threshold: float
    results_count: int
    processing_time_ms: int
    is_complete: bool


class ProgressiveSearchResponse(BaseModel):
    """Progressive search results."""
    stages: List[ProgressiveSearchStage]
    total_results: int
    total_time_ms: int
    results: List[Dict[str, Any]]
    completed_early: bool
    reason: Optional[str]


# ============================================================================
# Search Result Schemas
# ============================================================================

class SearchResult(BaseModel):
    """Individual search result."""
    photo_id: UUID
    face_id: UUID
    similarity_score: float
    thumbnail_url: Optional[str]
    person_id: Optional[UUID] = None
    person_name: Optional[str] = None
    
    # Context metadata
    photo_date: Optional[datetime] = None
    album_id: UUID
    album_name: Optional[str] = None
    
    # Quality metrics
    face_confidence: float
    face_quality: Optional[float] = None
    
    # Ranking
    rank: int
    relevance_score: Optional[float] = None


class SearchResponse(BaseModel):
    """Standard search response."""
    query_id: UUID
    total_results: int
    results: List[SearchResult]
    processing_time_ms: int
    search_params: Dict[str, Any]


class PersonSearchResult(BaseModel):
    """Person search result with photo."""
    photo_id: UUID
    album_id: UUID
    album_name: str
    face_ids: List[UUID]
    face_count: int
    thumbnail_url: Optional[str]
    photo_date: Optional[datetime]
    avg_quality: float


class CrossAlbumSearchResponse(BaseModel):
    """Cross-album search results."""
    query_id: UUID
    total_results: int
    albums_searched: int
    results_by_album: Dict[str, List[SearchResult]]  # album_id -> results
    top_results: List[SearchResult]  # Overall top results
    processing_time_ms: int


# ============================================================================
# Search Analytics Schemas
# ============================================================================

class SearchHistoryEntry(BaseModel):
    """Single search history entry."""
    id: UUID
    search_type: str
    query_params: Dict[str, Any]
    result_count: int
    created_at: datetime
    feedback_given: bool
    avg_relevance: Optional[float] = None


class SearchHistoryResponse(BaseModel):
    """User's search history."""
    total: int
    entries: List[SearchHistoryEntry]
    skip: int
    limit: int


class SearchFeedback(BaseModel):
    """Search result feedback."""
    search_id: UUID
    relevant_face_ids: List[UUID] = Field(default_factory=list)
    irrelevant_face_ids: List[UUID] = Field(default_factory=list)
    missing_expected: bool = Field(False, description="Expected results were missing")
    too_many_results: bool = Field(False, description="Too many irrelevant results")
    comments: Optional[str] = Field(None, max_length=500)


class SearchFeedbackResponse(BaseModel):
    """Feedback submission response."""
    message: str
    search_id: UUID
    feedback_recorded: bool
    improvements_applied: bool


class SearchImproveRequest(BaseModel):
    """Mark search results quality."""
    search_id: UUID
    result_corrections: List[Dict[str, Any]] = Field(
        ...,
        description="List of {face_id, is_correct, should_rank_higher/lower}"
    )
    
    @field_validator('result_corrections')
    def validate_corrections(cls, v: List[Dict[str, Any]]):
        for correction in v:
            if 'face_id' not in correction or 'is_correct' not in correction:
                raise ValueError("Each correction needs face_id and is_correct")
        return v


class SearchImproveResponse(BaseModel):
    """Search improvement response."""
    message: str
    corrections_applied: int
    model_updated: bool
    new_threshold_suggestion: Optional[float] = None


# ============================================================================
# Search Statistics Schemas
# ============================================================================

class SearchStatistics(BaseModel):
    """Search usage statistics."""
    total_searches: int
    avg_results_per_search: float
    most_used_search_type: str
    avg_processing_time_ms: int
    feedback_rate: float
    avg_relevance_score: Optional[float]
