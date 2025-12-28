"""
Production-Grade Audit Logging Service
======================================

Comprehensive audit logging system for tracking all user actions
with async support, database persistence, and background processing.

Features:
- Async and sync logging
- Database persistence
- Background task support
- IP address and user agent tracking
- JSON detail storage
- Batch logging support
- Error handling and fallback
- Query and filtering capabilities
"""

import logging
import json
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from datetime import datetime, timedelta
from contextlib import contextmanager

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_, desc

from src.db.base import SessionLocal
from src.models.audit_log import AuditLog
from src.models.enums import AuditAction


logger = logging.getLogger(__name__)


class AuditLogService:
    """
    Service for managing audit logs with database persistence.
    
    Features:
    - Synchronous and asynchronous logging
    - Automatic session management
    - Error handling with fallback logging
    - Batch operations
    - Query and filtering
    """
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize audit log service.
        
        Args:
            db: Optional database session. If not provided, creates new sessions.
        """
        self.db = db
        self._external_session = db is not None
    
    @contextmanager
    def _get_session(self):
        """Context manager for database session handling."""
        if self._external_session and self.db:
            yield self.db
        else:
            db = SessionLocal()
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()
    
    def record(
        self,
        user_id: Union[UUID, str],
        action: Union[str, AuditAction],
        resource_type: str,
        resource_id: Optional[Union[UUID, str]] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        commit: bool = True
    ) -> Optional[AuditLog]:
        """
        Record an audit log entry.
        
        Args:
            user_id: User who performed the action
            action: Action performed (create, read, update, delete, etc.)
            resource_type: Type of resource affected (person, album, photo, etc.)
            resource_id: ID of the affected resource
            details: Additional details as dictionary
            ip_address: Client IP address
            user_agent: Client user agent string
            commit: Whether to commit immediately (default: True)
        
        Returns:
            AuditLog instance if successful, None if failed
        
        Example:
            ```python
            audit_log.record(
                user_id=current_user.id,
                action="person.create",
                resource_type="person",
                resource_id=person.id,
                details={
                    "name": person.name,
                    "album_id": str(album.id)
                }
            )
            ```
        """
        try:
            with self._get_session() as db:
                # Convert action to enum if string
                if isinstance(action, str):
                    # Handle dot notation (e.g., "person.create" -> "create")
                    action_str = action.split('.')[-1] if '.' in action else action
                    try:
                        action_enum = AuditAction(action_str)
                    except ValueError:
                        logger.warning(f"Invalid audit action: {action_str}, defaulting to 'update'")
                        action_enum = AuditAction.update
                else:
                    action_enum = action
                
                # Convert UUIDs to proper format
                actor_id = UUID(str(user_id)) if user_id else None
                target_id = UUID(str(resource_id)) if resource_id else None
                
                # Serialize details to JSON
                details_json = json.dumps(details) if details else None
                
                # Create audit log entry
                audit_entry = AuditLog(
                    actor_id=actor_id,
                    action=action_enum,
                    target_type=resource_type,
                    target_id=target_id,
                    details=details_json,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                db.add(audit_entry)
                
                if commit and not self._external_session:
                    db.commit()
                    db.refresh(audit_entry)
                
                logger.debug(
                    f"Audit log recorded: {action_enum} on {resource_type} "
                    f"by user {user_id}"
                )
                
                return audit_entry
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to record audit log: {str(e)}", exc_info=True)
            # Fallback: log to file
            self._fallback_log(user_id, action, resource_type, resource_id, details)
            return None
        except Exception as e:
            logger.error(f"Unexpected error recording audit log: {str(e)}", exc_info=True)
            self._fallback_log(user_id, action, resource_type, resource_id, details)
            return None
    
    def record_batch(
        self,
        entries: List[Dict[str, Any]],
        commit: bool = True
    ) -> int:
        """
        Record multiple audit log entries in batch.
        
        Args:
            entries: List of audit log entry dictionaries
            commit: Whether to commit after batch insert
        
        Returns:
            Number of successfully recorded entries
        
        Example:
            ```python
            audit_log.record_batch([
                {
                    "user_id": user1.id,
                    "action": "create",
                    "resource_type": "person",
                    "resource_id": person1.id,
                    "details": {"name": "John"}
                },
                {
                    "user_id": user2.id,
                    "action": "update",
                    "resource_type": "person",
                    "resource_id": person2.id,
                    "details": {"name": "Jane"}
                }
            ])
            ```
        """
        success_count = 0
        
        try:
            with self._get_session() as db:
                for entry_data in entries:
                    try:
                        # Extract and convert data
                        action = entry_data.get('action', 'update')
                        if isinstance(action, str):
                            action_str = action.split('.')[-1] if '.' in action else action
                            try:
                                action_enum = AuditAction(action_str)
                            except ValueError:
                                action_enum = AuditAction.update
                        else:
                            action_enum = action
                        
                        user_id = entry_data.get('user_id')
                        resource_id = entry_data.get('resource_id')
                        details = entry_data.get('details')
                        
                        audit_entry = AuditLog(
                            actor_id=UUID(str(user_id)) if user_id else None,
                            action=action_enum,
                            target_type=entry_data.get('resource_type', 'unknown'),
                            target_id=UUID(str(resource_id)) if resource_id else None,
                            details=json.dumps(details) if details else None,
                            ip_address=entry_data.get('ip_address'),
                            user_agent=entry_data.get('user_agent')
                        )
                        
                        db.add(audit_entry)
                        success_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to add entry to batch: {str(e)}")
                        continue
                
                if commit and not self._external_session:
                    db.commit()
                
                logger.info(f"Batch recorded {success_count}/{len(entries)} audit logs")
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to record batch audit logs: {str(e)}", exc_info=True)
        
        return success_count
    
    def get_logs(
        self,
        user_id: Optional[UUID] = None,
        action: Optional[Union[str, AuditAction]] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100
    ) -> tuple[List[AuditLog], int]:
        """
        Query audit logs with filtering.
        
        Args:
            user_id: Filter by user who performed action
            action: Filter by action type
            resource_type: Filter by resource type
            resource_id: Filter by specific resource ID
            start_date: Filter logs from this date
            end_date: Filter logs until this date
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            Tuple of (logs list, total count)
        """
        try:
            with self._get_session() as db:
                query = db.query(AuditLog)
                
                # Apply filters
                if user_id:
                    query = query.filter(AuditLog.actor_id == user_id)
                
                if action:
                    if isinstance(action, str):
                        action_str = action.split('.')[-1] if '.' in action else action
                        try:
                            action_enum = AuditAction(action_str)
                            query = query.filter(AuditLog.action == action_enum)
                        except ValueError:
                            pass
                    else:
                        query = query.filter(AuditLog.action == action)
                
                if resource_type:
                    query = query.filter(AuditLog.target_type == resource_type)
                
                if resource_id:
                    query = query.filter(AuditLog.target_id == resource_id)
                
                if start_date:
                    query = query.filter(AuditLog.created_at >= start_date)
                
                if end_date:
                    query = query.filter(AuditLog.created_at <= end_date)
                
                # Get total count
                total = query.count()
                
                # Apply pagination and ordering
                logs = query.order_by(desc(AuditLog.created_at)).offset(skip).limit(limit).all()
                
                return logs, total
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to query audit logs: {str(e)}", exc_info=True)
            return [], 0
    
    def get_user_activity(
        self,
        user_id: UUID,
        days: int = 30,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Get recent activity for a specific user.
        
        Args:
            user_id: User ID to get activity for
            days: Number of days to look back
            limit: Maximum number of records
        
        Returns:
            List of audit logs
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        logs, _ = self.get_logs(
            user_id=user_id,
            start_date=start_date,
            limit=limit
        )
        return logs
    
    def get_resource_history(
        self,
        resource_type: str,
        resource_id: UUID,
        limit: int = 50
    ) -> List[AuditLog]:
        """
        Get complete history of a specific resource.
        
        Args:
            resource_type: Type of resource (person, album, etc.)
            resource_id: ID of the resource
            limit: Maximum number of records
        
        Returns:
            List of audit logs for the resource
        """
        logs, _ = self.get_logs(
            resource_type=resource_type,
            resource_id=resource_id,
            limit=limit
        )
        return logs
    
    def cleanup_old_logs(self, days: int = 90) -> int:
        """
        Delete audit logs older than specified days.
        
        Args:
            days: Delete logs older than this many days
        
        Returns:
            Number of deleted records
        """
        try:
            with self._get_session() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                deleted = db.query(AuditLog).filter(
                    AuditLog.created_at < cutoff_date
                ).delete(synchronize_session=False)
                
                if not self._external_session:
                    db.commit()
                
                logger.info(f"Cleaned up {deleted} audit logs older than {days} days")
                return deleted
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old audit logs: {str(e)}", exc_info=True)
            return 0
    
    def _fallback_log(
        self,
        user_id: Any,
        action: Any,
        resource_type: str,
        resource_id: Any,
        details: Optional[Dict[str, Any]]
    ):
        """
        Fallback logging to application logger when database fails.
        
        This ensures critical audit information is not lost even if
        the database is unavailable.
        """
        logger.warning(
            "AUDIT LOG (fallback): "
            f"user_id={user_id}, "
            f"action={action}, "
            f"resource_type={resource_type}, "
            f"resource_id={resource_id}, "
            f"details={json.dumps(details) if details else 'None'}"
        )


# =============================================================================
# Global Instance
# =============================================================================

# Global singleton instance for easy access
audit_log = AuditLogService()


# =============================================================================
# Async Helper Functions
# =============================================================================

async def record_audit_async(
    user_id: Union[UUID, str],
    action: Union[str, AuditAction],
    resource_type: str,
    resource_id: Optional[Union[UUID, str]] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> Optional[AuditLog]:
    """
    Async wrapper for recording audit logs.
    
    Can be used with BackgroundTasks for non-blocking logging.
    
    Example:
        ```python
        from fastapi import BackgroundTasks
        
        @router.post("/resource")
        async def create_resource(
            background_tasks: BackgroundTasks,
            current_user: User = Depends(get_current_user)
        ):
            # ... create resource ...
            
            background_tasks.add_task(
                record_audit_async,
                user_id=current_user.id,
                action="create",
                resource_type="resource",
                resource_id=resource.id
            )
        ```
    """
    try:
        return audit_log.record(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
    except Exception as e:
        logger.error(f"Async audit log failed: {str(e)}", exc_info=True)
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def record_login(user_id: UUID, ip_address: str, user_agent: str, success: bool = True):
    """Record user login attempt."""
    return audit_log.record(
        user_id=user_id,
        action=AuditAction.login,
        resource_type="user",
        resource_id=user_id,
        details={"success": success},
        ip_address=ip_address,
        user_agent=user_agent
    )


def record_logout(user_id: UUID, ip_address: str):
    """Record user logout."""
    return audit_log.record(
        user_id=user_id,
        action=AuditAction.logout,
        resource_type="user",
        resource_id=user_id,
        ip_address=ip_address
    )


def record_download(
    user_id: UUID,
    resource_type: str,
    resource_id: UUID,
    details: Optional[Dict[str, Any]] = None
):
    """Record resource download."""
    return audit_log.record(
        user_id=user_id,
        action=AuditAction.download,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details
    )


def record_share(
    user_id: UUID,
    resource_type: str,
    resource_id: UUID,
    shared_with: str,
    details: Optional[Dict[str, Any]] = None
):
    """Record resource sharing."""
    share_details = details or {}
    share_details['shared_with'] = shared_with
    
    return audit_log.record(
        user_id=user_id,
        action=AuditAction.share,
        resource_type=resource_type,
        resource_id=resource_id,
        details=share_details
    )
