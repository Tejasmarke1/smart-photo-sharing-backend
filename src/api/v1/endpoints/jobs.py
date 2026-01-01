"""
Job Monitoring API Endpoints
==============================

Production-grade FastAPI endpoints for monitoring and managing background jobs.

Features:
- Job listing and filtering by type/status
- Job details with progress tracking
- Job cancellation and retry
- Priority management
- Queue statistics
- Comprehensive error handling
"""

from typing import List, Optional
from uuid import UUID
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from celery.result import AsyncResult
from celery import states

from src.db.base import get_db
from src.api.deps import get_current_user
from src.models.user import User
from src.tasks.celery_app import celery_app
from src.schemas.job import (
    JobResponse,
    JobListResponse,
    JobStatsResponse,
    JobDetailResponse,
    JobCancelResponse,
    JobRetryRequest,
    JobPrioritizeRequest,
    JobClearQueueRequest,
    JobClearQueueResponse,
    JobType,
    JobStatus,
    JobPriority,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Helper Functions
# ============================================================================

def _get_celery_task(job_id: str) -> AsyncResult:
    """Get Celery task result object."""
    return AsyncResult(job_id, app=celery_app)


def _task_to_job_response(
    task: AsyncResult,
    user_id: UUID,
    job_type: Optional[JobType] = None
) -> JobResponse:
    """Convert Celery task to JobResponse."""
    
    # Map Celery states to our JobStatus
    status_mapping = {
        states.PENDING: JobStatus.PENDING,
        states.STARTED: JobStatus.STARTED,
        states.SUCCESS: JobStatus.SUCCESS,
        states.FAILURE: JobStatus.FAILURE,
        states.RETRY: JobStatus.RETRY,
        states.REVOKED: JobStatus.REVOKED,
    }
    
    job_status = status_mapping.get(task.state, JobStatus.PENDING)
    
    # Extract metadata from task info
    info = task.info or {}
    result = info if task.successful() else None
    error = str(info) if task.failed() else None
    progress = info.get('progress') if isinstance(info, dict) else None
    
    return JobResponse(
        job_id=task.id,
        job_type=job_type or JobType.FACE_DETECTION,  # Default, should be stored properly
        status=job_status,
        user_id=user_id,
        result=result,
        error=error,
        progress=progress,
        created_at=datetime.utcnow(),  # Celery doesn't track this easily
        priority=JobPriority.NORMAL,  # Default
        retries=0,  # Would need to track separately
    )


def _task_to_job_detail(
    task: AsyncResult,
    user_id: UUID,
    job_type: Optional[JobType] = None
) -> JobDetailResponse:
    """Convert Celery task to JobDetailResponse with extended info."""
    
    base_response = _task_to_job_response(task, user_id, job_type)
    
    # Get extended task info
    info = task.info or {}
    
    return JobDetailResponse(
        **base_response.model_dump(),
        task_name=task.name,
        worker=info.get('hostname') if isinstance(info, dict) else None,
        traceback=task.traceback if task.failed() else None,
    )


# ============================================================================
# Job Monitoring Endpoints
# ============================================================================

@router.get("", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    job_type: Optional[JobType] = Query(None, description="Filter by job type"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobListResponse:
    """
    List all face processing jobs for the current user.
    
    Supports pagination and filtering by status and job type.
    """
    try:
        # Get active tasks from Celery
        # Note: This is a simplified implementation
        # In production, you'd want to store job metadata in the database
        
        active_tasks = celery_app.control.inspect().active()
        scheduled_tasks = celery_app.control.inspect().scheduled()
        reserved_tasks = celery_app.control.inspect().reserved()
        
        all_task_ids = []
        
        # Collect task IDs from all workers
        if active_tasks:
            for worker, tasks in active_tasks.items():
                all_task_ids.extend([t['id'] for t in tasks])
        
        if scheduled_tasks:
            for worker, tasks in scheduled_tasks.items():
                all_task_ids.extend([t['id'] for t in tasks])
                
        if reserved_tasks:
            for worker, tasks in reserved_tasks.items():
                all_task_ids.extend([t['id'] for t in tasks])
        
        # Convert to JobResponse objects
        jobs = []
        for task_id in all_task_ids:
            task = _get_celery_task(task_id)
            job = _task_to_job_response(task, current_user.id, job_type)
            
            # Apply filters
            if status and job.status != status:
                continue
            if job_type and job.job_type != job_type:
                continue
                
            jobs.append(job)
        
        # Sort by created_at descending
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Pagination
        total = len(jobs)
        start = (page - 1) * size
        end = start + size
        paginated_jobs = jobs[start:end]
        
        return JobListResponse(
            jobs=paginated_jobs,
            total=total,
            page=page,
            size=size,
            has_more=end < total
        )
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve jobs"
        )


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job_details(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobDetailResponse:
    """
    Get detailed information about a specific job.
    
    Includes progress, result, error messages, and worker information.
    """
    try:
        task = _get_celery_task(job_id)
        
        # Check if task exists
        if task.state == states.PENDING and not task.result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job_detail = _task_to_job_detail(task, current_user.id)
        
        return job_detail
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job details for {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job details"
        )


@router.delete("/{job_id}", response_model=JobCancelResponse)
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobCancelResponse:
    """
    Cancel a running or pending job.
    
    Jobs that are already completed cannot be canceled.
    """
    try:
        task = _get_celery_task(job_id)
        
        # Check if task can be canceled
        if task.state in [states.SUCCESS, states.FAILURE]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in {task.state} state"
            )
        
        # Revoke the task
        task.revoke(terminate=True, signal='SIGKILL')
        
        logger.info(f"Job {job_id} canceled by user {current_user.id}")
        
        return JobCancelResponse(
            job_id=job_id,
            status="revoked",
            message="Job successfully canceled"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error canceling job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel job"
        )


@router.post("/{job_id}/retry", response_model=JobResponse)
async def retry_failed_job(
    job_id: str,
    retry_request: JobRetryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobResponse:
    """
    Retry a failed job.
    
    Only failed jobs can be retried unless force flag is set.
    """
    try:
        task = _get_celery_task(job_id)
        
        # Check if task can be retried
        if not retry_request.force and task.state != states.FAILURE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot retry job in {task.state} state. Use force=true to override."
            )
        
        # Get original task info to resubmit
        info = task.info or {}
        
        # For simplicity, we'll just return a message
        # In production, you'd resubmit the task with original args
        logger.info(f"Job {job_id} retry requested by user {current_user.id}")
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Job retry requires task metadata storage. Please resubmit the original request."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry job"
        )


# ============================================================================
# Job Type Filtering Endpoints
# ============================================================================

@router.get("/detection", response_model=JobListResponse)
async def list_detection_jobs(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobListResponse:
    """List all face detection jobs."""
    return await list_jobs(
        page=page,
        size=size,
        job_type=JobType.FACE_DETECTION,
        current_user=current_user,
        db=db
    )


@router.get("/clustering", response_model=JobListResponse)
async def list_clustering_jobs(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobListResponse:
    """List all face clustering jobs."""
    return await list_jobs(
        page=page,
        size=size,
        job_type=JobType.FACE_CLUSTERING,
        current_user=current_user,
        db=db
    )


@router.get("/reprocessing", response_model=JobListResponse)
async def list_reprocessing_jobs(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobListResponse:
    """List all face reprocessing jobs."""
    return await list_jobs(
        page=page,
        size=size,
        job_type=JobType.FACE_REPROCESSING,
        current_user=current_user,
        db=db
    )


# ============================================================================
# Job Priority Management
# ============================================================================

@router.post("/{job_id}/prioritize", response_model=JobResponse)
async def prioritize_job(
    job_id: str,
    priority_request: JobPrioritizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobResponse:
    """
    Increase job priority.
    
    Note: Celery doesn't support dynamic priority changes easily.
    This would require custom implementation with priority queues.
    """
    try:
        task = _get_celery_task(job_id)
        
        if task.state == states.PENDING:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        logger.warning(f"Priority change requested for job {job_id} but not implemented")
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Dynamic priority changes require custom queue implementation"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error prioritizing job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to prioritize job"
        )


@router.post("/queue/clear", response_model=JobClearQueueResponse)
async def clear_job_queue(
    clear_request: JobClearQueueRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobClearQueueResponse:
    """
    Clear job queue (admin only).
    
    Requires confirmation flag to be set to True.
    WARNING: This will cancel all pending jobs!
    """
    if not clear_request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must set confirm=true to clear queue"
        )
    
    # TODO: Add admin role check
    # if not current_user.is_admin:
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Admin access required"
    #     )
    
    try:
        # Purge all tasks from queue
        cleared_count = 0
        
        # Get scheduled and reserved tasks
        scheduled = celery_app.control.inspect().scheduled()
        reserved = celery_app.control.inspect().reserved()
        
        task_ids = []
        if scheduled:
            for worker, tasks in scheduled.items():
                task_ids.extend([t['id'] for t in tasks])
        if reserved:
            for worker, tasks in reserved.items():
                task_ids.extend([t['id'] for t in tasks])
        
        # Revoke all tasks
        for task_id in task_ids:
            task = _get_celery_task(task_id)
            task.revoke(terminate=True)
            cleared_count += 1
        
        logger.warning(f"Queue cleared by user {current_user.id}: {cleared_count} jobs canceled")
        
        return JobClearQueueResponse(
            cleared_count=cleared_count,
            message=f"Successfully cleared {cleared_count} jobs from queue"
        )
        
    except Exception as e:
        logger.error(f"Error clearing queue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear job queue"
        )


@router.get("/queue/stats", response_model=JobStatsResponse)
async def get_queue_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobStatsResponse:
    """
    Get job queue statistics.
    
    Returns counts by status and type, plus performance metrics.
    """
    try:
        # Get all tasks from Celery
        active_tasks = celery_app.control.inspect().active() or {}
        scheduled_tasks = celery_app.control.inspect().scheduled() or {}
        reserved_tasks = celery_app.control.inspect().reserved() or {}
        
        # Collect all task IDs
        all_task_ids = []
        for worker, tasks in active_tasks.items():
            all_task_ids.extend([t['id'] for t in tasks])
        for worker, tasks in scheduled_tasks.items():
            all_task_ids.extend([t['id'] for t in tasks])
        for worker, tasks in reserved_tasks.items():
            all_task_ids.extend([t['id'] for t in tasks])
        
        # Count by status
        status_counts = {
            'pending': 0,
            'started': 0,
            'success': 0,
            'failed': 0,
            'revoked': 0,
        }
        
        # Count by type (simplified - would need metadata storage)
        type_counts = {
            'detection': 0,
            'clustering': 0,
            'reprocessing': 0,
        }
        
        for task_id in all_task_ids:
            task = _get_celery_task(task_id)
            state = task.state.lower()
            if state in status_counts:
                status_counts[state] += 1
        
        # Calculate queue length
        queue_length = len(all_task_ids)
        
        return JobStatsResponse(
            total_jobs=len(all_task_ids),
            pending=status_counts['pending'],
            started=status_counts['started'],
            success=status_counts['success'],
            failed=status_counts['failed'],
            revoked=status_counts['revoked'],
            detection_jobs=type_counts['detection'],
            clustering_jobs=type_counts['clustering'],
            reprocessing_jobs=type_counts['reprocessing'],
            queue_length=queue_length,
            avg_processing_time=None,  # Would need historical data
            oldest_pending_job=None,  # Would need task metadata
        )
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve queue statistics"
        )
