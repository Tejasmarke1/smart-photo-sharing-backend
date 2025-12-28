"""
Person Management Celery Workers
=================================

Async tasks for heavy person operations with progress tracking,
error recovery, and comprehensive logging.

Tasks:
- Batch merge persons
- Batch label faces
- Transfer person between albums
- Cleanup operations
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import logging
import time
import traceback

from celery import Task, states
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.tasks.celery_app import celery_app
from src.db.base import SessionLocal
from src.models.person import Person
from src.models.face import Face
from src.models.face_person import FacePerson
from src.repositories.person_repo import PersonRepository
from src.repositories.album_repo import AlbumRepository
from src.core.audit_log import audit_log
from src.services.webhooks import trigger_webhook
from src.core.cache import invalidate_cache

logger = logging.getLogger(__name__)


# =============================================================================
# Base Task Class
# =============================================================================

class PersonTask(Task):
    """Base task for person operations."""
    
    autoretry_for = (SQLAlchemyError,)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    
    def get_db(self) -> Session:
        """Get database session."""
        return SessionLocal()
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(
            f"Task {task_id} failed: {str(exc)}",
            extra={
                "task_id": task_id,
                "args": args,
                "kwargs": kwargs,
                "traceback": traceback.format_exc()
            }
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(
            f"Task {task_id} completed successfully",
            extra={
                "task_id": task_id,
                "result": retval
            }
        )


# =============================================================================
# Batch Merge Task
# =============================================================================

@celery_app.task(
    bind=True,
    base=PersonTask,
    name='tasks.batch_merge_persons',
    track_started=True
)
def batch_merge_task(
    self,
    person_ids: List[str],
    target_id: Optional[str] = None,
    merged_name: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Merge multiple persons into one (async).
    
    Args:
        person_ids: List of person UUIDs to merge
        target_id: Target person UUID (if None, uses first)
        merged_name: Optional name for merged person
        user_id: User performing the operation
        
    Returns:
        Merge summary with statistics
        
    Progress:
        0-20%: Validation
        20-80%: Merging persons
        80-100%: Cache invalidation & cleanup
    """
    db = self.get_db()
    start_time = time.time()
    
    try:
        # Update state: Starting
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Validating persons',
                'progress': 10,
                'current': 0,
                'total': len(person_ids)
            }
        )
        
        logger.info(f"Starting batch merge of {len(person_ids)} persons")
        
        # Determine target
        target_uuid = UUID(target_id) if target_id else UUID(person_ids[0])
        person_uuids = [UUID(pid) for pid in person_ids]
        
        # Validate
        repo = PersonRepository(db)
        persons = [repo.get_by_id(pid) for pid in person_uuids]
        
        if any(p is None for p in persons):
            raise ValueError("One or more persons not found")
        
        # Check same album
        album_ids = {p.album_id for p in persons}
        if len(album_ids) > 1:
            raise ValueError("Cannot merge persons from different albums")
        
        album_id = persons[0].album_id
        
        # Update state: Merging
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Merging persons',
                'progress': 30,
                'current': 0,
                'total': len(person_ids)
            }
        )
        
        # Count total faces before merge
        total_faces = sum(repo.get_face_count(pid) for pid in person_uuids)
        
        # Perform merge one by one
        merged_person = None
        for i, pid in enumerate(person_uuids):
            if pid == target_uuid:
                merged_person = repo.get_by_id(pid)
                continue
            
            try:
                merged_person = repo.merge_persons(
                    source_id=pid,
                    target_id=target_uuid,
                    keep_source_name=False
                )
                
                # Update progress
                progress = 30 + int((i + 1) / len(person_uuids) * 50)
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'status': f'Merged {i+1}/{len(person_uuids)} persons',
                        'progress': progress,
                        'current': i + 1,
                        'total': len(person_uuids)
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to merge person {pid}: {str(e)}")
                # Continue with remaining persons
                continue
        
        # Update name if specified
        if merged_name and merged_person:
            merged_person.name = merged_name
        
        db.commit()
        
        if not merged_person:
            raise ValueError("Merge operation failed")
        
        # Update state: Cleanup
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Cleaning up',
                'progress': 85
            }
        )
        
        # Invalidate cache
        for pid in person_uuids:
            try:
                invalidate_cache(f"person:details:{pid}")
                invalidate_cache(f"person:faces:{pid}")
            except Exception as e:
                logger.warning(f"Failed to invalidate cache: {e}")
        
        invalidate_cache(f"album:persons:{album_id}")
        
        # Audit log
        if user_id:
            audit_log.record(
                user_id=UUID(user_id),
                action="person.batch_merge_completed",
                resource_type="person",
                resource_id=merged_person.id,
                details={
                    "source_count": len(person_ids),
                    "total_faces": total_faces,
                    "processing_time": time.time() - start_time
                }
            )
        
        # Trigger webhook
        try:
            trigger_webhook(
                event="person.batch_merged",
                data={
                    "merged_person_id": str(merged_person.id),
                    "person_count": len(person_ids),
                    "total_faces": total_faces,
                    "album_id": str(album_id)
                }
            )
        except Exception as e:
            logger.warning(f"Webhook trigger failed: {e}")
        
        processing_time = time.time() - start_time
        
        result = {
            'status': 'completed',
            'merged_person_id': str(merged_person.id),
            'merged_person_name': merged_person.name,
            'persons_merged': len(person_ids),
            'total_faces': total_faces,
            'album_id': str(album_id),
            'processing_time': processing_time
        }
        
        logger.info(
            f"Batch merge completed successfully",
            extra=result
        )
        
        return result
        
    except Exception as e:
        db.rollback()
        logger.error(f"Batch merge failed: {str(e)}", exc_info=True)
        
        self.update_state(
            state=states.FAILURE,
            meta={
                'status': 'Failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )
        
        raise
        
    finally:
        db.close()


# =============================================================================
# Batch Label Task
# =============================================================================

@celery_app.task(
    bind=True,
    base=PersonTask,
    name='tasks.batch_label_faces',
    track_started=True
)
def batch_label_task(
    self,
    face_ids: List[str],
    person_id: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Label multiple faces with a person (async).
    
    Args:
        face_ids: List of face UUIDs to label
        person_id: Person UUID to assign faces to
        user_id: User performing the operation
        
    Returns:
        Labeling summary with statistics
    """
    db = self.get_db()
    start_time = time.time()
    
    try:
        # Update state: Starting
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Validating faces',
                'progress': 10,
                'current': 0,
                'total': len(face_ids)
            }
        )
        
        logger.info(f"Starting batch label of {len(face_ids)} faces")
        
        person_uuid = UUID(person_id)
        face_uuids = [UUID(fid) for fid in face_ids]
        
        # Validate person exists
        repo = PersonRepository(db)
        person = repo.get_by_id(person_uuid)
        
        if not person:
            raise ValueError(f"Person {person_id} not found")
        
        # Batch label faces
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Labeling faces',
                'progress': 30
            }
        )
        
        labeled_count = 0
        failed_faces = []
        
        for i, face_id in enumerate(face_uuids):
            try:
                # Check if face already labeled
                existing = db.query(FacePerson).filter(
                    FacePerson.face_id == face_id
                ).first()
                
                if existing:
                    # Update existing mapping
                    existing.person_id = person_uuid
                    existing.is_manual = True
                    existing.confidence = 1.0
                else:
                    # Create new mapping
                    mapping = FacePerson(
                        face_id=face_id,
                        person_id=person_uuid,
                        is_manual=True,
                        confidence=1.0
                    )
                    db.add(mapping)
                
                labeled_count += 1
                
                # Commit every 100 faces
                if (i + 1) % 100 == 0:
                    db.commit()
                
                # Update progress
                progress = 30 + int((i + 1) / len(face_ids) * 60)
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'status': f'Labeled {i+1}/{len(face_ids)} faces',
                        'progress': progress,
                        'current': i + 1,
                        'total': len(face_ids)
                    }
                )
                
            except Exception as e:
                logger.warning(f"Failed to label face {face_id}: {str(e)}")
                failed_faces.append(str(face_id))
                continue
        
        # Final commit
        db.commit()
        
        # Update state: Cleanup
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Cleaning up',
                'progress': 95
            }
        )
        
        # Invalidate cache
        invalidate_cache(f"person:faces:{person_uuid}")
        invalidate_cache(f"album:persons:{person.album_id}")
        
        # Audit log
        if user_id:
            audit_log.record(
                user_id=UUID(user_id),
                action="person.batch_label_completed",
                resource_type="person",
                resource_id=person_uuid,
                details={
                    "faces_labeled": labeled_count,
                    "failed_count": len(failed_faces),
                    "processing_time": time.time() - start_time
                }
            )
        
        processing_time = time.time() - start_time
        
        result = {
            'status': 'completed',
            'person_id': str(person_uuid),
            'person_name': person.name,
            'faces_labeled': labeled_count,
            'failed_faces': failed_faces,
            'processing_time': processing_time
        }
        
        logger.info(
            f"Batch label completed",
            extra=result
        )
        
        return result
        
    except Exception as e:
        db.rollback()
        logger.error(f"Batch label failed: {str(e)}", exc_info=True)
        
        self.update_state(
            state=states.FAILURE,
            meta={
                'status': 'Failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )
        
        raise
        
    finally:
        db.close()


# =============================================================================
# Transfer Person Task
# =============================================================================

@celery_app.task(
    bind=True,
    base=PersonTask,
    name='tasks.transfer_person',
    track_started=True
)
def transfer_person_task(
    self,
    person_id: str,
    target_album_id: str,
    merge_if_exists: bool = False,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transfer person between albums (async).
    
    Args:
        person_id: Person UUID to transfer
        target_album_id: Target album UUID
        merge_if_exists: Merge with existing person if name matches
        user_id: User performing operation
        
    Returns:
        Transfer summary
    """
    db = self.get_db()
    start_time = time.time()
    
    try:
        # Update state: Starting
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Validating',
                'progress': 10
            }
        )
        
        logger.info(f"Starting person transfer {person_id} to album {target_album_id}")
        
        person_uuid = UUID(person_id)
        target_album_uuid = UUID(target_album_id)
        
        # Validate
        repo = PersonRepository(db)
        album_repo = AlbumRepository(db)
        
        person = repo.get_by_id(person_uuid)
        if not person:
            raise ValueError(f"Person {person_id} not found")
        
        target_album = album_repo.get_by_id(target_album_uuid)
        if not target_album:
            raise ValueError(f"Album {target_album_id} not found")
        
        source_album_id = person.album_id
        
        # Count faces
        face_count = repo.get_face_count(person_uuid)
        
        # Update state: Transferring
        self.update_state(
            state='PROCESSING',
            meta={
                'status': f'Transferring {face_count} faces',
                'progress': 30
            }
        )
        
        # Perform transfer
        result = repo.transfer_to_album(
            person_uuid,
            target_album_uuid,
            merge_if_exists
        )
        
        if not result:
            raise ValueError("Transfer operation failed")
        
        transferred_person, was_merged, merged_with_id = result
        
        db.commit()
        
        # Update state: Cleanup
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Cleaning up',
                'progress': 90
            }
        )
        
        # Invalidate cache
        invalidate_cache(f"person:details:{person_uuid}")
        invalidate_cache(f"album:persons:{source_album_id}")
        invalidate_cache(f"album:persons:{target_album_uuid}")
        
        # Audit log
        if user_id:
            audit_log.record(
                user_id=UUID(user_id),
                action="person.transfer_completed",
                resource_type="person",
                resource_id=person_uuid,
                details={
                    "source_album": str(source_album_id),
                    "target_album": str(target_album_uuid),
                    "was_merged": was_merged,
                    "face_count": face_count,
                    "processing_time": time.time() - start_time
                }
            )
        
        processing_time = time.time() - start_time
        
        result_data = {
            'status': 'completed',
            'person_id': str(transferred_person.id),
            'person_name': transferred_person.name,
            'source_album_id': str(source_album_id),
            'target_album_id': str(target_album_uuid),
            'faces_transferred': face_count,
            'was_merged': was_merged,
            'merged_with_id': str(merged_with_id) if merged_with_id else None,
            'processing_time': processing_time
        }
        
        logger.info(
            f"Person transfer completed",
            extra=result_data
        )
        
        return result_data
        
    except Exception as e:
        db.rollback()
        logger.error(f"Person transfer failed: {str(e)}", exc_info=True)
        
        self.update_state(
            state=states.FAILURE,
            meta={
                'status': 'Failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )
        
        raise
        
    finally:
        db.close()


# =============================================================================
# Cleanup Task
# =============================================================================

@celery_app.task(
    bind=True,
    base=PersonTask,
    name='tasks.cleanup_orphaned_persons'
)
def cleanup_orphaned_persons_task(self) -> Dict[str, Any]:
    """
    Clean up persons with no face mappings.
    
    Returns:
        Cleanup summary
    """
    db = self.get_db()
    
    try:
        logger.info("Starting orphaned persons cleanup")
        
        # Find persons with no faces
        repo = PersonRepository(db)
        orphaned = repo.find_orphaned_persons()
        
        deleted_count = 0
        
        for person in orphaned:
            try:
                repo.delete(person.id, force=True)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete person {person.id}: {str(e)}")
                continue
        
        db.commit()
        
        result = {
            'status': 'completed',
            'orphaned_persons_deleted': deleted_count
        }
        
        logger.info(f"Cleaned up {deleted_count} orphaned persons")
        
        return result
        
    except Exception as e:
        db.rollback()
        logger.error(f"Cleanup failed: {str(e)}", exc_info=True)
        raise
        
    finally:
        db.close()


# =============================================================================
# Scheduled Tasks
# =============================================================================

@celery_app.task(name='tasks.scheduled_cleanup')
def scheduled_cleanup_task():
    """
    Scheduled cleanup of orphaned persons (runs daily).
    """
    return cleanup_orphaned_persons_task.delay()