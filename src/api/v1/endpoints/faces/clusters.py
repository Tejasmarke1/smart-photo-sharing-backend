"""
Face Detection and Search API Endpoints
========================================

Production-grade FastAPI endpoints for face detection, labeling, and search.

Features:
- Face detection and embedding generation
- Face labeling and person management
- Selfie-based search
- Album-level clustering
- Quality assessment and filtering
- Comprehensive error handling
- Rate limiting and caching
- OpenAPI documentation
"""

from typing import List, Optional, Dict, Any
import json
from uuid import UUID
import logging
from datetime import datetime

from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    status, 
    UploadFile, 
    File,
    Query,
    Body,
    BackgroundTasks
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import numpy as np
import cv2
from pydantic import BaseModel, Field, validator

from src.db.base import get_db
from src.models.face import Face
from src.models.face_person import FacePerson
from src.models.person import Person
from src.models.photo import Photo
from src.models.album import Album
from src.models.face_cluster import FaceCluster
from src.api.deps import get_current_user, require_roles
from src.models.user import User
from src.services.face.pipeline import FacePipeline, create_pipeline
from src.services.storage.s3 import S3Service
from src.schemas.face import (
    FaceResponse,
    FaceLabelRequest,
    FaceSearchRequest,
    FaceSearchResponse,
    FaceClusterRequest,
    FaceQualityFilter,
    FaceListResult,
    JobAccepted,
    ReprocessResponse,
    BatchReprocessRequest,
    BatchDeleteRequest,
    DuplicateFace,
    OutlierFace,
    FaceQualityResponse,
    AlbumDetectRequest,
    AdvancedFilterRequest,
    QualityCheckRequest,
    AutoClusterRequest,
    ClusterStatusResponse,
    ClusterDetailResponse,
    ClusterResultsResponse,
    ClusterReviewRequest,
    ClusterAcceptRequest,
    ClusterSplitRequest,
    ClusterMergeRequest,
    AutoLabelRequest,
    LabelSuggestion,
    LabelSuggestionsResponse,
    ConfirmLabelRequest,
    AutoLabelResponse
)
from src.tasks.workers.face_processor import process_faces_task, process_album_photos_task, cluster_album_task
from src.core.cache import cache_result, invalidate_cache
from src.core.rate_limiter import rate_limit
from src.utils.validators import validate_image

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()


# Shared helpers for faces endpoints
from src.api.v1.endpoints.faces.helpers import get_pipeline, get_face_or_404, serialize_face

router = APIRouter()

@router.post(
    "/albums/{album_id}/cluster",
    response_model=Dict[str, Any],
    summary="Cluster faces in album",
    description="Run clustering to group similar faces (find same people)",
    status_code=status.HTTP_202_ACCEPTED
)
@require_roles(['photographer', 'admin','guest'])
async def cluster_album(
    album_id: UUID,
    request: FaceClusterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Trigger face clustering for an album.
    
    This is an async operation that:
    1. Groups similar faces together
    2. Suggests person clusters
    3. Identifies potential merge candidates
    
    Returns a job ID for status tracking.
    """
    # Verify album ownership
    album = db.query(Album).filter(Album.id == album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {album_id} not found"
        )
    
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Check if album has faces
    face_count = db.query(func.count(Face.id)).join(Photo).filter(
        Photo.album_id == album_id
    ).scalar()
    
    if face_count < request.min_cluster_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Album has only {face_count} faces, need at least {request.min_cluster_size}"
        )
    
    # Enqueue clustering task
    task_id = cluster_album_task.delay(
        str(album_id),
        min_cluster_size=request.min_cluster_size,
        similarity_threshold=request.similarity_threshold,
        merge_threshold=request.merge_threshold
    )
    
    logger.info(f"Enqueued clustering task {task_id} for album {album_id}")
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "message": "Clustering job started",
            "job_id": str(task_id),
            "album_id": str(album_id),
            "face_count": face_count,
            "status_url": f"/api/v1/jobs/{task_id}"
        }
    )


@router.post(
    "/albums/{album_id}/cluster/auto",
    response_model=ClusterStatusResponse,
    summary="Auto-cluster with smart defaults",
    description="Automatically cluster faces using intelligent algorithm parameters",
    status_code=status.HTTP_202_ACCEPTED
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="auto_cluster", max_calls=5, period=300)
async def auto_cluster_album(
    album_id: UUID,
    request: AutoClusterRequest = Body(default_factory=AutoClusterRequest),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline)
):
    """
    Auto-cluster faces with smart defaults.
    
    Smart defaults:
    - Automatically determine optimal cluster size
    - Adaptive similarity threshold based on data
    - Intelligent noise filtering
    
    Returns job ID for status tracking.
    """
    # Verify album ownership
    album = db.query(Album).filter(Album.id == album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {album_id} not found"
        )
    
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Count faces
    face_count = db.query(func.count(Face.id)).join(Photo).filter(
        Photo.album_id == album_id,
        Face.embedding.isnot(None)
    ).scalar()
    
    if face_count < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Album has only {face_count} faces with embeddings, need at least 3"
        )
    
    # Determine parameters
    if request.use_smart_defaults:
        # Smart parameter selection based on dataset size
        if face_count < 20:
            min_cluster_size = 2
            similarity_threshold = 0.65
        elif face_count < 100:
            min_cluster_size = 3
            similarity_threshold = 0.70
        else:
            min_cluster_size = 5
            similarity_threshold = 0.75
    else:
        min_cluster_size = request.min_cluster_size or 3
        similarity_threshold = request.similarity_threshold or 0.70
    
    # Enqueue clustering task
    from src.tasks.workers.face_processor import cluster_album_task
    task_id = cluster_album_task.delay(
        str(album_id),
        min_cluster_size=min_cluster_size,
        similarity_threshold=similarity_threshold,
        merge_threshold=0.85
    )
    
    logger.info(f"Enqueued auto-clustering task {task_id} for album {album_id} ({face_count} faces)")
    
    return ClusterStatusResponse(
        job_id=str(task_id),
        status="pending",
        total_faces=face_count,
        started_at=datetime.utcnow()
    )


@router.get(
    "/albums/{album_id}/cluster/status",
    response_model=ClusterStatusResponse,
    summary="Get clustering job status",
    description="Check the status of a clustering job"
)
async def get_clustering_status(
    album_id: UUID,
    job_id: str = Query(..., description="Clustering job ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get clustering job status.
    
    Status values:
    - pending: Job queued
    - running: Processing
    - completed: Done
    - failed: Error occurred
    """
    # Verify album access
    album = db.query(Album).filter(Album.id == album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {album_id} not found"
        )
    
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Check Celery task status
    from celery.result import AsyncResult
    task = AsyncResult(job_id)
    
    # Get cluster count if completed
    num_clusters = None
    if task.state == 'SUCCESS':
        clusters = db.query(FaceCluster).filter(
            FaceCluster.album_id == album_id,
            FaceCluster.job_id == job_id
        ).count()
        num_clusters = clusters
    
    return ClusterStatusResponse(
        job_id=job_id,
        status=task.state.lower(),
        progress=None,  # Could add progress tracking
        num_clusters=num_clusters,
        error=str(task.info) if task.state == 'FAILURE' else None
    )


@router.get(
    "/albums/{album_id}/cluster/results",
    response_model=ClusterResultsResponse,
    summary="Get clustering results",
    description="Retrieve complete clustering results for review"
)
@cache_result(ttl=300)
async def get_clustering_results(
    album_id: UUID,
    job_id: Optional[str] = Query(None, description="Specific job ID, or latest if not provided"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Get clustering results for review.
    
    Returns:
    - All clusters with sample faces
    - Noise faces (not clustered)
    - Merge suggestions
    - Cluster quality metrics
    """
    # Verify album access
    album = db.query(Album).filter(Album.id == album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {album_id} not found"
        )
    
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get clusters
    query = db.query(FaceCluster).filter(FaceCluster.album_id == album_id)
    if job_id:
        query = query.filter(FaceCluster.job_id == job_id)
    
    clusters = query.order_by(FaceCluster.size.desc()).all()
    
    if not clusters:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No clustering results found"
        )
    
    # Build cluster details
    cluster_details = []
    all_clustered_face_ids = set()
    
    for cluster in clusters:
        face_ids = cluster.face_ids if isinstance(cluster.face_ids, list) else []
        all_clustered_face_ids.update(face_ids)
        
        # Get sample thumbnails (first 6)
        sample_faces = db.query(Face).filter(Face.id.in_(face_ids[:6])).all()
        sample_thumbnails = []
        
        for face in sample_faces:
            if face.thumbnail_s3_key:
                url = s3_service.generate_presigned_download_url(
                    face.thumbnail_s3_key, expires_in=3600
                )
                sample_thumbnails.append(url)
        
        # Get representative thumbnail
        rep_thumbnail = None
        if cluster.representative_face_id:
            rep_face = db.query(Face).filter(Face.id == cluster.representative_face_id).first()
            if rep_face and rep_face.thumbnail_s3_key:
                rep_thumbnail = s3_service.generate_presigned_download_url(
                    rep_face.thumbnail_s3_key, expires_in=3600
                )
        
        cluster_details.append(ClusterDetailResponse(
            id=cluster.id,
            album_id=cluster.album_id,
            cluster_label=cluster.cluster_label,
            size=cluster.size,
            status=cluster.status,
            avg_similarity=cluster.avg_similarity,
            confidence_score=cluster.confidence_score,
            representative_face_id=cluster.representative_face_id,
            representative_thumbnail_url=rep_thumbnail,
            face_ids=[UUID(fid) for fid in face_ids],
            sample_thumbnails=sample_thumbnails,
            person_id=cluster.person_id,
            person_name=cluster.person.name if cluster.person else None,
            created_at=cluster.created_at
        ))
    
    # Find noise faces (not in any cluster)
    total_faces = db.query(func.count(Face.id)).join(Photo).filter(
        Photo.album_id == album_id,
        Face.embedding.isnot(None)
    ).scalar()
    
    all_face_ids = db.query(Face.id).join(Photo).filter(
        Photo.album_id == album_id,
        Face.embedding.isnot(None)
    ).all()
    
    noise_faces = [fid[0] for fid in all_face_ids if str(fid[0]) not in all_clustered_face_ids]
    
    # Get merge suggestions (clusters with high similarity)
    merge_suggestions = []
    # TODO: Implement merge suggestion algorithm
    
    return ClusterResultsResponse(
        job_id=clusters[0].job_id or "unknown",
        album_id=album_id,
        total_faces=total_faces,
        num_clusters=len(clusters),
        clusters=cluster_details,
        noise_faces=noise_faces,
        merge_suggestions=merge_suggestions,
        completed_at=clusters[0].created_at
    )


@router.post(
    "/albums/{album_id}/cluster/review",
    response_model=Dict[str, Any],
    summary="Submit cluster review",
    description="Review and apply corrections to clustering results"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="cluster_review", max_calls=20, period=60)
async def submit_cluster_review(
    album_id: UUID,
    request: ClusterReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submit cluster review with corrections.
    
    Actions:
    - accept: Convert cluster to person
    - reject: Mark as noise
    - split: Divide into multiple clusters
    - merge: Combine with other clusters
    """
    # Verify album access
    album = db.query(Album).filter(Album.id == album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {album_id} not found"
        )
    
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    results = {
        'accepted': 0,
        'rejected': 0,
        'split': 0,
        'merged': 0,
        'errors': []
    }
    
    for review in request.reviews:
        cluster_id = UUID(review['cluster_id'])
        action = review['action']
        data = review.get('data', {})
        
        cluster = db.query(FaceCluster).filter(
            FaceCluster.id == cluster_id,
            FaceCluster.album_id == album_id
        ).first()
        
        if not cluster:
            results['errors'].append(f"Cluster {cluster_id} not found")
            continue
        
        try:
            if action == 'accept':
                # Create person from cluster
                person_name = data.get('person_name', f'Person {cluster.cluster_label}')
                person = Person(
                    album_id=album_id,
                    name=person_name,
                    representative_face_id=cluster.representative_face_id
                )
                db.add(person)
                db.flush()
                
                # Link faces to person
                face_ids = cluster.face_ids if isinstance(cluster.face_ids, list) else []
                for face_id_str in face_ids:
                    mapping = FacePerson(
                        face_id=UUID(face_id_str),
                        person_id=person.id,
                        is_manual=False
                    )
                    db.add(mapping)
                
                cluster.status = 'accepted'
                cluster.person_id = person.id
                results['accepted'] += 1
                
            elif action == 'reject':
                cluster.status = 'rejected'
                results['rejected'] += 1
                
            elif action == 'split':
                # Mark for split (actual split handled separately)
                cluster.status = 'split'
                results['split'] += 1
                
            elif action == 'merge':
                # Mark for merge (actual merge handled separately)
                cluster.status = 'merged'
                results['merged'] += 1
            
            cluster.reviewed_by_user_id = user_id
            cluster.review_notes = data.get('notes')
            
        except Exception as e:
            logger.error(f"Error processing review for cluster {cluster_id}: {e}")
            results['errors'].append(f"Cluster {cluster_id}: {str(e)}")
    
    db.commit()
    
    return {
        "message": "Review submitted successfully",
        "results": results
    }


# ============================================================================
# Cluster Operations Endpoints
# ============================================================================

@router.get(
    "/clusters/{cluster_id}",
    response_model=ClusterDetailResponse,
    summary="Get cluster details",
    description="Retrieve detailed information about a specific cluster"
)
async def get_cluster_details(
    cluster_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """Get detailed information about a cluster."""
    cluster = db.query(FaceCluster).filter(FaceCluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster {cluster_id} not found"
        )
    
    # Verify album access
    album = db.query(Album).filter(Album.id == cluster.album_id).first()
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get sample thumbnails
    face_ids = cluster.face_ids if isinstance(cluster.face_ids, list) else []
    sample_faces = db.query(Face).filter(Face.id.in_(face_ids[:6])).all()
    sample_thumbnails = []
    
    for face in sample_faces:
        if face.thumbnail_s3_key:
            url = s3_service.generate_presigned_download_url(
                face.thumbnail_s3_key, expires_in=3600
            )
            sample_thumbnails.append(url)
    
    # Get representative thumbnail
    rep_thumbnail = None
    if cluster.representative_face_id:
        rep_face = db.query(Face).filter(Face.id == cluster.representative_face_id).first()
        if rep_face and rep_face.thumbnail_s3_key:
            rep_thumbnail = s3_service.generate_presigned_download_url(
                rep_face.thumbnail_s3_key, expires_in=3600
            )
    
    return ClusterDetailResponse(
        id=cluster.id,
        album_id=cluster.album_id,
        cluster_label=cluster.cluster_label,
        size=cluster.size,
        status=cluster.status,
        avg_similarity=cluster.avg_similarity,
        confidence_score=cluster.confidence_score,
        representative_face_id=cluster.representative_face_id,
        representative_thumbnail_url=rep_thumbnail,
        face_ids=[UUID(fid) for fid in face_ids],
        sample_thumbnails=sample_thumbnails,
        person_id=cluster.person_id,
        person_name=cluster.person.name if cluster.person else None,
        created_at=cluster.created_at
    )


@router.post(
    "/clusters/{cluster_id}/accept",
    response_model=Dict[str, Any],
    summary="Accept cluster as person",
    description="Convert cluster to a person record"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="accept_cluster", max_calls=50, period=60)
async def accept_cluster(
    cluster_id: UUID,
    request: ClusterAcceptRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Accept cluster and create person.
    
    Process:
    1. Create person record
    2. Link all faces to person
    3. Mark cluster as accepted
    """
    cluster = db.query(FaceCluster).filter(FaceCluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster {cluster_id} not found"
        )
    
    # Verify album access
    album = db.query(Album).filter(Album.id == cluster.album_id).first()
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    if cluster.status == 'accepted':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cluster already accepted"
        )
    
    # Create person
    person = Person(
        album_id=cluster.album_id,
        name=request.person_name,
        email=request.person_email,
        phone=request.person_phone,
        representative_face_id=cluster.representative_face_id
    )
    db.add(person)
    db.flush()
    
    # Link faces to person
    face_ids = cluster.face_ids if isinstance(cluster.face_ids, list) else []
    linked_count = 0
    
    for face_id_str in face_ids:
        # Check if already linked
        existing = db.query(FacePerson).filter(
            FacePerson.face_id == UUID(face_id_str)
        ).first()
        
        if not existing:
            mapping = FacePerson(
                face_id=UUID(face_id_str),
                person_id=person.id,
                is_manual=False
            )
            db.add(mapping)
            linked_count += 1
    
    # Update cluster
    cluster.status = 'accepted'
    cluster.person_id = person.id
    cluster.reviewed_by_user_id = user_id
    
    db.commit()
    
    logger.info(f"Accepted cluster {cluster_id} as person {person.id} ({linked_count} faces)")
    
    return {
        "message": "Cluster accepted successfully",
        "person_id": str(person.id),
        "person_name": person.name,
        "faces_linked": linked_count
    }


@router.post(
    "/clusters/{cluster_id}/reject",
    response_model=Dict[str, Any],
    summary="Reject cluster",
    description="Mark cluster as noise/invalid"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="reject_cluster", max_calls=50, period=60)
async def reject_cluster(
    cluster_id: UUID,
    notes: Optional[str] = Body(None, description="Rejection reason"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Reject cluster as noise/invalid.
    
    Rejected clusters won't be suggested again.
    Faces remain available for other operations.
    """
    cluster = db.query(FaceCluster).filter(FaceCluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster {cluster_id} not found"
        )
    
    # Verify album access
    album = db.query(Album).filter(Album.id == cluster.album_id).first()
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    cluster.status = 'rejected'
    cluster.reviewed_by_user_id = user_id
    cluster.review_notes = notes
    
    db.commit()
    
    logger.info(f"Rejected cluster {cluster_id}")
    
    return {
        "message": "Cluster rejected successfully",
        "cluster_id": str(cluster_id)
    }


@router.post(
    "/clusters/{cluster_id}/split",
    response_model=Dict[str, Any],
    summary="Split cluster",
    description="Split a cluster into multiple sub-clusters"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="split_cluster", max_calls=20, period=60)
async def split_cluster(
    cluster_id: UUID,
    request: ClusterSplitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Split cluster into multiple clusters.
    
    Useful when algorithm incorrectly grouped different people.
    """
    cluster = db.query(FaceCluster).filter(FaceCluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster {cluster_id} not found"
        )
    
    # Verify album access
    album = db.query(Album).filter(Album.id == cluster.album_id).first()
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Validate face groups
    original_face_ids = set(cluster.face_ids if isinstance(cluster.face_ids, list) else [])
    provided_face_ids = set()
    for group in request.face_groups:
        provided_face_ids.update(str(fid) for fid in group)
    
    if provided_face_ids != original_face_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Face groups must contain all faces from original cluster"
        )
    
    # Get max cluster label
    max_label = db.query(func.max(FaceCluster.cluster_label)).filter(
        FaceCluster.album_id == cluster.album_id
    ).scalar() or 0
    
    # Create new clusters
    new_clusters = []
    for i, group in enumerate(request.face_groups):
        new_cluster = FaceCluster(
            album_id=cluster.album_id,
            job_id=cluster.job_id,
            cluster_label=max_label + i + 1,
            size=len(group),
            status='pending',
            face_ids=[str(fid) for fid in group],
            representative_face_id=group[0]  # Use first face as representative
        )
        db.add(new_cluster)
        new_clusters.append(new_cluster)
    
    # Mark original as split
    cluster.status = 'split'
    cluster.reviewed_by_user_id = user_id
    
    db.commit()
    
    logger.info(f"Split cluster {cluster_id} into {len(new_clusters)} clusters")
    
    return {
        "message": "Cluster split successfully",
        "original_cluster_id": str(cluster_id),
        "new_cluster_ids": [str(c.id) for c in new_clusters],
        "split_count": len(new_clusters)
    }


@router.post(
    "/clusters/merge",
    response_model=Dict[str, Any],
    summary="Merge clusters",
    description="Merge multiple clusters into one"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="merge_clusters", max_calls=20, period=60)
async def merge_clusters(
    request: ClusterMergeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Merge multiple clusters into one.
    
    Useful when algorithm split same person into multiple clusters.
    """
    if len(request.cluster_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 clusters to merge"
        )
    
    # Fetch clusters
    clusters = db.query(FaceCluster).filter(
        FaceCluster.id.in_(request.cluster_ids)
    ).all()
    
    if len(clusters) != len(request.cluster_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Some clusters not found"
        )
    
    # Verify all from same album
    album_ids = set(c.album_id for c in clusters)
    if len(album_ids) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All clusters must be from same album"
        )
    
    # Verify album access
    album = db.query(Album).filter(Album.id == list(album_ids)[0]).first()
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Collect all face IDs
    all_face_ids = []
    total_size = 0
    for cluster in clusters:
        face_ids = cluster.face_ids if isinstance(cluster.face_ids, list) else []
        all_face_ids.extend(face_ids)
        total_size += cluster.size
    
    # Create merged cluster
    merged_cluster = FaceCluster(
        album_id=album.id,
        job_id=clusters[0].job_id,
        cluster_label=clusters[0].cluster_label,  # Use first cluster's label
        size=total_size,
        status='pending',
        face_ids=all_face_ids,
        representative_face_id=clusters[0].representative_face_id
    )
    db.add(merged_cluster)
    db.flush()
    
    # Mark originals as merged
    for cluster in clusters:
        cluster.status = 'merged'
        cluster.merged_into_cluster_id = merged_cluster.id
        cluster.reviewed_by_user_id = user_id
    
    db.commit()
    
    logger.info(f"Merged {len(clusters)} clusters into {merged_cluster.id}")
    
    return {
        "message": "Clusters merged successfully",
        "merged_cluster_id": str(merged_cluster.id),
        "original_cluster_ids": [str(c.id) for c in clusters],
        "total_faces": total_size
    }


# ============================================================================
# Auto-labeling Endpoints
# ============================================================================

@router.post(
    "/faces/auto-label",
    response_model=AutoLabelResponse,
    summary="Auto-label faces",
    description="Automatically label faces based on existing persons",
    status_code=status.HTTP_202_ACCEPTED
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="auto_label", max_calls=10, period=60)
async def auto_label_faces(
    request: AutoLabelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline)
):
    """
    Auto-label faces using existing persons.
    
    Process:
    1. Get unlabeled faces from album
    2. For each person, find similar faces
    3. Auto-label if similarity above threshold
    
    Returns count of labeled faces.
    """
    # Verify album access
    album = db.query(Album).filter(Album.id == request.album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {request.album_id} not found"
        )
    
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get persons to use as reference
    person_query = db.query(Person).filter(Person.album_id == request.album_id)
    if request.person_ids:
        person_query = person_query.filter(Person.id.in_(request.person_ids))
    
    persons = person_query.all()
    
    if not persons:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No persons found for auto-labeling"
        )
    
    # Get unlabeled faces
    faces_query = db.query(Face).join(Photo).filter(
        Photo.album_id == request.album_id,
        Face.embedding.isnot(None)
    )
    
    if request.unlabeled_only:
        faces_query = faces_query.outerjoin(FacePerson).filter(FacePerson.face_id.is_(None))
    
    faces = faces_query.limit(request.max_faces).all()
    
    if not faces:
        return AutoLabelResponse(
            message="No unlabeled faces found",
            labeled_count=0,
            skipped_count=0,
            face_ids=[],
            person_assignments={}
        )
    
    # Auto-label
    labeled_count = 0
    skipped_count = 0
    labeled_faces = []
    assignments = {}
    
    for face in faces:
        face_emb = np.array(face.embedding, dtype=np.float32)
        best_match = None
        best_score = 0.0
        
        # Find best matching person
        for person in persons:
            # Get person's face embeddings
            person_faces = db.query(Face).join(FacePerson).filter(
                FacePerson.person_id == person.id,
                Face.embedding.isnot(None)
            ).all()
            
            if not person_faces:
                continue
            
            # Calculate average similarity
            person_embs = np.array([f.embedding for f in person_faces], dtype=np.float32)
            similarities = np.dot(person_embs, face_emb) / (
                np.linalg.norm(person_embs, axis=1) * np.linalg.norm(face_emb)
            )
            avg_similarity = float(np.mean(similarities))
            
            if avg_similarity > best_score:
                best_score = avg_similarity
                best_match = person
        
        # Label if above threshold
        if best_match and best_score >= request.min_confidence:
            mapping = FacePerson(
                face_id=face.id,
                person_id=best_match.id,
                is_manual=False
            )
            db.add(mapping)
            labeled_count += 1
            labeled_faces.append(face.id)
            assignments[str(face.id)] = best_match.id
        else:
            skipped_count += 1
    
    db.commit()
    
    logger.info(f"Auto-labeled {labeled_count} faces in album {request.album_id}")
    
    return AutoLabelResponse(
        message=f"Auto-labeled {labeled_count} faces",
        labeled_count=labeled_count,
        skipped_count=skipped_count,
        face_ids=labeled_faces,
        person_assignments=assignments
    )


@router.get(
    "/faces/label-suggestions",
    response_model=LabelSuggestionsResponse,
    summary="Get labeling suggestions",
    description="Get AI-powered suggestions for unlabeled faces"
)
@cache_result(ttl=300)
async def get_label_suggestions(
    album_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    min_confidence: float = Query(0.75, ge=0.6, le=0.95),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    pipeline: FacePipeline = Depends(get_pipeline),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """
    Get labeling suggestions for unlabeled faces.
    
    Returns:
    - Face
    - Suggested person
    - Confidence level
    - Reasoning
    """
    # Verify album access
    album = db.query(Album).filter(Album.id == album_id).first()
    if not album:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Album {album_id} not found"
        )
    
    user_id = getattr(current_user, 'id', None)
    role = getattr(current_user, 'role', None)
    if isinstance(current_user, dict):
        user_id = user_id or current_user.get('id')
        role = role or current_user.get('role')
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    if isinstance(user_id, str):
        try:
            user_id = UUID(user_id)
        except ValueError:
            pass
    
    if album.photographer_id != user_id and role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get persons
    persons = db.query(Person).filter(Person.album_id == album_id).all()
    
    if not persons:
        return LabelSuggestionsResponse(
            album_id=album_id,
            suggestions=[],
            total=0
        )
    
    # Get unlabeled faces
    faces = db.query(Face).join(Photo).outerjoin(FacePerson).filter(
        Photo.album_id == album_id,
        Face.embedding.isnot(None),
        FacePerson.face_id.is_(None)
    ).limit(limit).all()
    
    suggestions = []
    
    for face in faces:
        face_emb = np.array(face.embedding, dtype=np.float32)
        best_match = None
        best_score = 0.0
        
        # Find best matching person
        for person in persons:
            person_faces = db.query(Face).join(FacePerson).filter(
                FacePerson.person_id == person.id,
                Face.embedding.isnot(None)
            ).all()
            
            if not person_faces:
                continue
            
            person_embs = np.array([f.embedding for f in person_faces], dtype=np.float32)
            similarities = np.dot(person_embs, face_emb) / (
                np.linalg.norm(person_embs, axis=1) * np.linalg.norm(face_emb)
            )
            avg_similarity = float(np.mean(similarities))
            
            if avg_similarity > best_score:
                best_score = avg_similarity
                best_match = person
        
        # Add suggestion if above threshold
        if best_match and best_score >= min_confidence:
            thumbnail_url = None
            if face.thumbnail_s3_key:
                thumbnail_url = s3_service.generate_presigned_download_url(
                    face.thumbnail_s3_key, expires_in=3600
                )
            
            # Determine confidence level
            if best_score >= 0.9:
                confidence = "high"
                reasoning = f"Very high similarity ({best_score:.2%}) to known faces"
            elif best_score >= 0.8:
                confidence = "medium"
                reasoning = f"Good similarity ({best_score:.2%}) to known faces"
            else:
                confidence = "low"
                reasoning = f"Moderate similarity ({best_score:.2%}), review recommended"
            
            suggestions.append(LabelSuggestion(
                face_id=face.id,
                photo_id=face.photo_id,
                thumbnail_url=thumbnail_url,
                person_id=best_match.id,
                person_name=best_match.name,
                similarity_score=best_score,
                confidence=confidence,
                reasoning=reasoning
            ))
    
    return LabelSuggestionsResponse(
        album_id=album_id,
        suggestions=suggestions,
        total=len(suggestions)
    )


@router.post(
    "/faces/{face_id}/confirm-label",
    response_model=FaceResponse,
    summary="Confirm auto-label",
    description="Accept and apply suggested label"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="confirm_label", max_calls=100, period=60)
async def confirm_face_label(
    face_id: UUID,
    request: ConfirmLabelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    s3_service: S3Service = Depends(lambda: S3Service())
):
    """Confirm and apply suggested label."""
    face = await get_face_or_404(face_id, db, current_user)
    
    # Check if already labeled
    existing = db.query(FacePerson).filter(FacePerson.face_id == face_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Face already labeled"
        )
    
    # Verify person exists
    person = db.query(Person).filter(Person.id == request.person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found"
        )
    
    # Create mapping
    mapping = FacePerson(
        face_id=face_id,
        person_id=request.person_id,
        is_manual=True  # User confirmed
    )
    db.add(mapping)
    db.commit()
    
    logger.info(f"Confirmed label for face {face_id} -> person {request.person_id}")
    
    return serialize_face(face, s3_service)


@router.post(
    "/faces/{face_id}/reject-label",
    response_model=Dict[str, Any],
    summary="Reject auto-label",
    description="Reject suggested label"
)
@require_roles(['photographer', 'admin'])
@rate_limit(key_prefix="reject_label", max_calls=100, period=60)
async def reject_face_label(
    face_id: UUID,
    reason: Optional[str] = Body(None, description="Rejection reason"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Reject suggested label.
    
    This helps improve future suggestions by learning from rejections.
    """
    face = await get_face_or_404(face_id, db, current_user)
    
    # TODO: Store rejection for ML feedback
    logger.info(f"Rejected label suggestion for face {face_id}: {reason}")
    
    return {
        "message": "Label suggestion rejected",
        "face_id": str(face_id)
    }


# ============================================================================
# Error Handlers
# ============================================================================


