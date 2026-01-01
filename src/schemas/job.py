"""
Job Schemas
============

Pydantic schemas for background job monitoring and management.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class JobType(str, Enum):
    """Types of background jobs."""
    FACE_DETECTION = "face_detection"
    FACE_CLUSTERING = "face_clustering"
    FACE_REPROCESSING = "face_reprocessing"
    PERSON_MERGING = "person_merging"
    SEARCH_INDEXING = "search_indexing"
    BATCH_UPLOAD = "batch_upload"


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class JobBase(BaseModel):
    """Base job schema."""
    job_id: str = Field(..., description="Unique job identifier")
    job_type: JobType
    status: JobStatus
    created_at: datetime


class JobResponse(JobBase):
    """Job response with full details."""
    user_id: UUID
    result: Optional[Dict[str, Any]] = Field(None, description="Job result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Progress percentage")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = Field(0, ge=0)
    priority: JobPriority = JobPriority.NORMAL
    
    # Related resource IDs
    photo_id: Optional[UUID] = None
    album_id: Optional[UUID] = None
    person_id: Optional[UUID] = None
    
    model_config = ConfigDict(from_attributes=True)


class JobListResponse(BaseModel):
    """Paginated list of jobs."""
    jobs: List[JobResponse]
    total: int
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1, le=100)
    has_more: bool


class JobStatsResponse(BaseModel):
    """Job queue statistics."""
    total_jobs: int
    pending: int
    started: int
    success: int
    failed: int
    revoked: int
    
    # By type
    detection_jobs: int
    clustering_jobs: int
    reprocessing_jobs: int
    
    # Performance metrics
    avg_processing_time: Optional[float] = Field(None, description="Average time in seconds")
    oldest_pending_job: Optional[datetime] = None
    queue_length: int = Field(0, ge=0)


class JobRetryRequest(BaseModel):
    """Request to retry a failed job."""
    force: bool = Field(False, description="Force retry even if not failed")


class JobPrioritizeRequest(BaseModel):
    """Request to increase job priority."""
    priority: JobPriority = Field(..., description="New priority level")


class JobCancelResponse(BaseModel):
    """Response after canceling a job."""
    job_id: str
    status: str
    message: str


class JobClearQueueRequest(BaseModel):
    """Request to clear job queue."""
    job_types: Optional[List[JobType]] = Field(None, description="Specific job types to clear")
    confirm: bool = Field(..., description="Confirmation flag - must be True")


class JobClearQueueResponse(BaseModel):
    """Response after clearing queue."""
    cleared_count: int
    message: str


class JobDetailResponse(JobResponse):
    """Extended job details with metadata."""
    task_name: Optional[str] = None
    task_args: Optional[List[Any]] = None
    task_kwargs: Optional[Dict[str, Any]] = None
    worker: Optional[str] = Field(None, description="Worker that processed the job")
    traceback: Optional[str] = Field(None, description="Full traceback if failed")
    eta: Optional[datetime] = Field(None, description="Estimated time of arrival")
    expires: Optional[datetime] = Field(None, description="Expiration time")
    
    model_config = ConfigDict(from_attributes=True)
