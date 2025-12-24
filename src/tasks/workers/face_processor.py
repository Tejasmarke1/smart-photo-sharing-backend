"""
Face Processing Celery Workers
===============================

Asynchronous tasks for face detection, embedding, and clustering.

Features:
- Idempotent task design
- Automatic retries with exponential backoff
- Progress tracking
- Error handling and logging
- Resource cleanup
"""

from typing import List, Dict, Any, Optional
import json
from uuid import UUID
import logging
import traceback
import time

import numpy as np
import cv2
from celery import Task, group, chain
from sqlalchemy.orm import Session

from src.tasks.celery_app import celery_app
from src.db.base import SessionLocal
from src.models.face import Face
from src.models.person import Person
from src.models.photo import Photo
from src.models.album import Album
from src.services.face.pipeline import create_pipeline, FacePipeline
from src.services.storage.s3 import S3Service
from src.core.cache import invalidate_cache

logger = logging.getLogger(__name__)


# ============================================================================
# Base Task Class
# ============================================================================

class FaceTask(Task):
    """Base task class with common functionality."""
    
    # Retry configuration
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    
    # Shared pipeline instance per worker
    _pipeline: Optional[FacePipeline] = None
    _s3_service: Optional[S3Service] = None
    
    @property
    def pipeline(self) -> FacePipeline:
        """Lazy-load face pipeline."""
        if self._pipeline is None:
            logger.info("Initializing face pipeline in worker")
            # Create pipeline without FAISS to avoid training errors with few faces
            self._pipeline = create_pipeline(enable_search=False)
        return self._pipeline
    
    @property
    def s3_service(self) -> S3Service:
        """Lazy-load S3 service."""
        if self._s3_service is None:
            self._s3_service = S3Service()
        return self._s3_service
    
    def get_db(self) -> Session:
        """Get database session."""
        return SessionLocal()


# ============================================================================
# Face Detection & Embedding Task
# ============================================================================

@celery_app.task(
    bind=True,
    base=FaceTask,
    name='tasks.process_faces',
    track_started=True
)
def process_faces_task(
    self,
    photo_id: str,
    force_reprocess: bool = False
) -> Dict[str, Any]:
    """
    Process photo: detect faces, generate embeddings, save thumbnails.
    
    Args:
        photo_id: Photo UUID
        force_reprocess: If True, delete existing faces and reprocess
        
    Returns:
        Processing summary
        
    Workflow:
        1. Download photo from S3
        2. Detect faces
        3. Align faces
        4. Generate embeddings
        5. Save face thumbnails to S3
        6. Store faces in database
        7. Add embeddings to vector index
    """
    db = self.get_db()
    start_time = time.time()
    
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Loading photo', 'progress': 10}
        )
        
        # Get photo
        photo = db.query(Photo).filter(Photo.id == UUID(photo_id)).first()
        if not photo:
            raise ValueError(f"Photo {photo_id} not found")
        
        logger.info(f"Processing faces for photo {photo_id}")
        
        # Check if already processed
        if not force_reprocess:
            existing_faces = db.query(Face).filter(Face.photo_id == UUID(photo_id)).count()
            if existing_faces > 0:
                logger.info(f"Photo {photo_id} already has {existing_faces} faces, skipping")
                return {
                    'status': 'skipped',
                    'photo_id': photo_id,
                    'faces_count': existing_faces,
                    'message': 'Already processed'
                }
        else:
            # Delete existing faces
            db.query(Face).filter(Face.photo_id == UUID(photo_id)).delete()
            db.commit()
        
        # Download image from S3
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Downloading image', 'progress': 20}
        )
        
        image_bytes = self.s3_service.download_file(photo.s3_key)
        if not image_bytes:
            raise ValueError(f"Failed to download photo from {photo.s3_key}")
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process photo with pipeline
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Detecting faces', 'progress': 40}
        )
        
        face_results = self.pipeline.process_photo(
            photo_id=photo_id,
            image=image_rgb,
            save_crops=False  # We'll handle saving separately
        )
        
        if not face_results:
            logger.info(f"No faces detected in photo {photo_id}")
            return {
                'status': 'completed',
                'photo_id': photo_id,
                'faces_count': 0,
                'processing_time': time.time() - start_time
            }
        
        logger.info(f"Detected {len(face_results)} faces in photo {photo_id}")
        
        # Save face crops and create DB records
        self.update_state(
            state='PROCESSING',
            meta={
                'status': f'Saving {len(face_results)} faces',
                'progress': 60
            }
        )
        
        saved_faces = []
        
        for i, result in enumerate(face_results):
            try:
                # Extract face crop
                x, y, w, h = result.bbox
                face_crop = image_rgb[y:y+h, x:x+w]
                
                # Resize to standard size
                face_crop_resized = cv2.resize(face_crop, (224, 224))
                
                # Convert to BGR for saving
                face_crop_bgr = cv2.cvtColor(face_crop_resized, cv2.COLOR_RGB2BGR)
                
                # Encode as JPEG
                success, buffer = cv2.imencode('.jpg', face_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    logger.warning(f"Failed to encode face {i} for photo {photo_id}")
                    continue
                
                # Upload to S3
                s3_key = f"faces/{photo_id}/{result.face_id}.jpg"
                upload_success = self.s3_service.upload_file(
                    buffer.tobytes(),
                    s3_key,
                    content_type='image/jpeg'
                )
                
                if not upload_success:
                    logger.warning(f"Failed to upload face crop to {s3_key}")
                    s3_key = None
                
                # Create Face record
                bbox_dict = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

                face = Face(
                    id=UUID(result.face_id),
                    photo_id=UUID(photo_id),
                    bbox=json.dumps(bbox_dict),
                    embedding=result.embedding.tolist(),
                    confidence=float(result.confidence),
                    thumbnail_s3_key=s3_key,
                    blur_score=float(result.blur_score),
                    brightness_score=float(result.brightness_score)
                )
                
                db.add(face)
                saved_faces.append(face)
                
                # Update progress
                progress = 60 + int((i + 1) / len(face_results) * 30)
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'status': f'Saved face {i+1}/{len(face_results)}',
                        'progress': progress
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to save face {i} for photo {photo_id}: {str(e)}")
                continue
        
        # Commit to database
        db.commit()
        
        logger.info(f"Saved {len(saved_faces)} faces for photo {photo_id}")
        
        # Invalidate cache
        photo_obj = db.query(Photo).filter(Photo.id == UUID(photo_id)).first()
        if photo_obj:
            invalidate_cache(f"album_faces_{photo_obj.album_id}")
        
        processing_time = time.time() - start_time
        
        return {
            'status': 'completed',
            'photo_id': photo_id,
            'faces_detected': len(face_results),
            'faces_saved': len(saved_faces),
            'processing_time': processing_time
        }
        
    except Exception as e:
        logger.error(f"Face processing failed for photo {photo_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update task state
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )
        
        raise
        
    finally:
        db.close()


# ============================================================================
# Batch Photo Processing
# ============================================================================

@celery_app.task(
    bind=True,
    base=FaceTask,
    name='tasks.process_album_photos'
)
def process_album_photos_task(
    self,
    album_id: str,
    photo_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process all photos in an album (or specific photos).
    
    Args:
        album_id: Album UUID
        photo_ids: Optional list of specific photo IDs
        
    Returns:
        Processing summary
    """
    db = self.get_db()
    
    try:
        # Get photos
        query = db.query(Photo).filter(Photo.album_id == UUID(album_id))
        
        if photo_ids:
            query = query.filter(Photo.id.in_([UUID(pid) for pid in photo_ids]))
        
        photos = query.all()
        
        if not photos:
            return {
                'status': 'completed',
                'album_id': album_id,
                'photos_processed': 0,
                'message': 'No photos to process'
            }
        
        logger.info(f"Processing {len(photos)} photos for album {album_id}")
        
        # Create group of tasks
        job = group(
            process_faces_task.s(str(photo.id))
            for photo in photos
        )
        
        # Execute
        result = job.apply_async()
        
        # Wait for completion (optional, can be made async)
        results = result.get()
        
        # Aggregate results
        total_faces = sum(r.get('faces_saved', 0) for r in results if r)
        
        return {
            'status': 'completed',
            'album_id': album_id,
            'photos_processed': len(photos),
            'total_faces': total_faces,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Album photo processing failed: {str(e)}")
        raise
        
    finally:
        db.close()


# ============================================================================
# Face Clustering Task
# ============================================================================

@celery_app.task(
    bind=True,
    base=FaceTask,
    name='tasks.cluster_album',
    track_started=True
)
def cluster_album_task(
    self,
    album_id: str,
    min_cluster_size: int = 5,
    similarity_threshold: float = 0.7,
    merge_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Cluster faces in an album to identify unique people.
    
    Args:
        album_id: Album UUID
        min_cluster_size: Minimum faces per cluster
        similarity_threshold: Threshold for initial clustering
        merge_threshold: Threshold for suggesting cluster merges
        
    Returns:
        Clustering summary with person suggestions
        
    Workflow:
        1. Load all faces with embeddings
        2. Run clustering algorithm
        3. Suggest person clusters
        4. Identify merge candidates
        5. (Optional) Auto-create person entities
    """
    db = self.get_db()
    start_time = time.time()
    
    try:
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Loading faces', 'progress': 10}
        )
        
        # Get album
        album = db.query(Album).filter(Album.id == UUID(album_id)).first()
        if not album:
            raise ValueError(f"Album {album_id} not found")
        
        logger.info(f"Clustering faces for album {album_id}")
        
        # Load all faces with embeddings
        faces = db.query(Face).join(Photo).filter(
            Photo.album_id == UUID(album_id),
            Face.embedding.isnot(None)
        ).all()
        
        if len(faces) < min_cluster_size:
            return {
                'status': 'skipped',
                'album_id': album_id,
                'face_count': len(faces),
                'message': f'Not enough faces (minimum {min_cluster_size})'
            }
        
        logger.info(f"Loaded {len(faces)} faces for clustering")
        
        # Prepare data
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Preparing data', 'progress': 30}
        )
        
        embeddings = np.vstack([np.array(face.embedding) for face in faces])
        face_ids = [str(face.id) for face in faces]
        photo_ids = [str(face.photo_id) for face in faces]
        
        # Run clustering
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Running clustering', 'progress': 50}
        )
        
        cluster_map = self.pipeline.clusterer.cluster(
            embeddings,
            face_ids,
            photo_ids,
            min_cluster_size=min_cluster_size,
            eps=1 - similarity_threshold  # Convert similarity to distance
        )
        
        # Get unique clusters
        unique_clusters = set(cluster_map.values())
        unique_clusters.discard(-1)  # Remove noise cluster
        
        logger.info(f"Found {len(unique_clusters)} clusters")
        
        # Suggest merges
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Analyzing clusters', 'progress': 70}
        )
        
        merge_suggestions = self.pipeline.clusterer.suggest_merges(
            cluster_map,
            embeddings,
            face_ids,
            threshold=merge_threshold
        )
        
        logger.info(f"Generated {len(merge_suggestions)} merge suggestions")
        
        # Create cluster summary
        cluster_summary = {}
        for cluster_id in unique_clusters:
            cluster_face_ids = [
                fid for fid, cid in cluster_map.items()
                if cid == cluster_id
            ]
            
            cluster_summary[str(cluster_id)] = {
                'face_count': len(cluster_face_ids),
                'face_ids': cluster_face_ids,
                'suggested_name': f"Person {cluster_id}"
            }
        
        # Store clustering results (optional)
        # Could store in a separate table for UI display
        
        processing_time = time.time() - start_time
        
        return {
            'status': 'completed',
            'album_id': album_id,
            'total_faces': len(faces),
            'num_clusters': len(unique_clusters),
            'noise_faces': list(cluster_map.values()).count(-1),
            'clusters': cluster_summary,
            'merge_suggestions': merge_suggestions,
            'processing_time': processing_time
        }
        
    except Exception as e:
        logger.error(f"Clustering failed for album {album_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )
        
        raise
        
    finally:
        db.close()


# ============================================================================
# Auto-Labeling Task
# ============================================================================

@celery_app.task(
    bind=True,
    base=FaceTask,
    name='tasks.auto_label_cluster'
)
def auto_label_cluster_task(
    self,
    album_id: str,
    cluster_id: int,
    person_name: str
) -> Dict[str, Any]:
    """
    Auto-label all faces in a cluster as a specific person.
    
    Args:
        album_id: Album UUID
        cluster_id: Cluster ID from clustering results
        person_name: Name for the person
        
    Returns:
        Labeling summary
    """
    db = self.get_db()
    
    try:
        # Get or create person
        person = db.query(Person).filter(
            Person.album_id == UUID(album_id),
            Person.name == person_name
        ).first()
        
        if not person:
            person = Person(
                album_id=UUID(album_id),
                name=person_name
            )
            db.add(person)
            db.flush()
        
        # Get faces in cluster (this would require storing cluster results)
        # For now, this is a placeholder
        
        # Create face-person mappings
        # ... mapping logic ...
        
        db.commit()
        
        return {
            'status': 'completed',
            'album_id': album_id,
            'cluster_id': cluster_id,
            'person_id': str(person.id),
            'faces_labeled': 0  # Update with actual count
        }
        
    except Exception as e:
        logger.error(f"Auto-labeling failed: {str(e)}")
        db.rollback()
        raise
        
    finally:
        db.close()


# ============================================================================
# Embedding Recomputation (Model Updates)
# ============================================================================

@celery_app.task(
    bind=True,
    base=FaceTask,
    name='tasks.recompute_embeddings'
)
def recompute_embeddings_task(
    self,
    album_id: Optional[str] = None,
    face_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Recompute embeddings for faces (useful after model updates).
    
    Args:
        album_id: Optional album to recompute
        face_ids: Optional specific faces to recompute
        
    Returns:
        Recomputation summary
    """
    db = self.get_db()
    
    try:
        # Build query
        query = db.query(Face)
        
        if album_id:
            query = query.join(Photo).filter(Photo.album_id == UUID(album_id))
        
        if face_ids:
            query = query.filter(Face.id.in_([UUID(fid) for fid in face_ids]))
        
        faces = query.all()
        
        logger.info(f"Recomputing embeddings for {len(faces)} faces")
        
        updated_count = 0
        
        for face in faces:
            try:
                # Download face crop
                if not face.thumbnail_s3_key:
                    continue
                
                image_bytes = self.s3_service.download_file(face.thumbnail_s3_key)
                if not image_bytes:
                    continue
                
                # Decode
                nparr = np.frombuffer(image_bytes, np.uint8)
                face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Compute new embedding
                new_embedding = self.pipeline.embedder.embed(face_img_rgb)
                
                # Update
                face.embedding = new_embedding.tolist()
                updated_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to recompute embedding for face {face.id}: {str(e)}")
                continue
        
        db.commit()
        
        logger.info(f"Recomputed {updated_count} embeddings")
        
        return {
            'status': 'completed',
            'faces_processed': len(faces),
            'embeddings_updated': updated_count
        }
        
    except Exception as e:
        logger.error(f"Embedding recomputation failed: {str(e)}")
        db.rollback()
        raise
        
    finally:
        db.close()


# ============================================================================
# Cleanup Task
# ============================================================================

@celery_app.task(
    bind=True,
    name='tasks.cleanup_orphaned_faces'
)
def cleanup_orphaned_faces_task(self) -> Dict[str, Any]:
    """
    Clean up orphaned face records (no associated photo).
    
    Returns:
        Cleanup summary
    """
    db = SessionLocal()
    
    try:
        # Find orphaned faces
        orphaned = db.query(Face).outerjoin(Photo).filter(
            Photo.id.is_(None)
        ).all()
        
        logger.info(f"Found {len(orphaned)} orphaned faces")
        
        # Delete thumbnails from S3
        s3_service = S3Service()
        for face in orphaned:
            if face.thumbnail_s3_key:
                try:
                    s3_service.delete_file(face.thumbnail_s3_key)
                except Exception as e:
                    logger.warning(f"Failed to delete {face.thumbnail_s3_key}: {str(e)}")
        
        # Delete from DB
        for face in orphaned:
            db.delete(face)
        
        db.commit()
        
        return {
            'status': 'completed',
            'orphaned_faces_deleted': len(orphaned)
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        db.rollback()
        raise
        
    finally:
        db.close()