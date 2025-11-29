"""Base repository with common CRUD operations."""
from typing import TypeVar, Generic, Type, Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, or_
from uuid import UUID
from datetime import datetime

from src.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with common database operations."""
    
    def __init__(self, model: Type[ModelType], db: Session):
        """
        Initialize repository.
        
        Args:
            model: SQLAlchemy model class
            db: Database session
        """
        self.model = model
        self.db = db
    
    def create(self, obj_in: Dict[str, Any]) -> ModelType:
        """
        Create new record.
        
        Args:
            obj_in: Dictionary with object data
            
        Returns:
            Created model instance
        """
        db_obj = self.model(**obj_in)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def get(self, id: UUID, include_deleted: bool = False) -> Optional[ModelType]:
        """
        Get record by ID.
        
        Args:
            id: Record UUID
            include_deleted: Whether to include soft-deleted records
            
        Returns:
            Model instance or None if not found
        """
        query = self.db.query(self.model).filter(self.model.id == id)
        
        # Handle soft delete if model has deleted_at column
        if hasattr(self.model, 'deleted_at') and not include_deleted:
            query = query.filter(self.model.deleted_at.is_(None))
        
        return query.first()
    
    def get_multi(
        self,
        skip: int = 0,
        limit: int = 100,
        include_deleted: bool = False,
        order_by: Optional[str] = None,
        order_desc: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[ModelType], int]:
        """
        Get multiple records with pagination and filtering.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            include_deleted: Whether to include soft-deleted records
            order_by: Column name to order by (defaults to 'created_at')
            order_desc: Whether to order descending
            filters: Dictionary of column:value filters
            
        Returns:
            Tuple of (list of records, total count)
        """
        query = self.db.query(self.model)
        
        # Apply soft delete filter
        if hasattr(self.model, 'deleted_at') and not include_deleted:
            query = query.filter(self.model.deleted_at.is_(None))
        
        # Apply additional filters
        if filters:
            for column, value in filters.items():
                if hasattr(self.model, column):
                    query = query.filter(getattr(self.model, column) == value)
        
        # Get total count
        total = query.count()
        
        # Apply ordering
        if order_by and hasattr(self.model, order_by):
            order_column = getattr(self.model, order_by)
        elif hasattr(self.model, 'created_at'):
            order_column = self.model.created_at
        else:
            order_column = self.model.id
        
        if order_desc:
            query = query.order_by(desc(order_column))
        else:
            query = query.order_by(asc(order_column))
        
        # Apply pagination
        records = query.offset(skip).limit(limit).all()
        
        return records, total
    
    def update(self, id: UUID, obj_in: Dict[str, Any]) -> Optional[ModelType]:
        """
        Update record.
        
        Args:
            id: Record UUID
            obj_in: Dictionary with fields to update
            
        Returns:
            Updated model instance or None if not found
        """
        db_obj = self.get(id)
        if not db_obj:
            return None
        
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def delete(self, id: UUID, soft: bool = True) -> bool:
        """
        Delete record (soft or hard).
        
        Args:
            id: Record UUID
            soft: Whether to soft delete (if model supports it)
            
        Returns:
            True if successful, False if record not found
        """
        db_obj = self.get(id, include_deleted=True)
        if not db_obj:
            return False
        
        # Soft delete if supported and requested
        if soft and hasattr(self.model, 'deleted_at'):
            db_obj.deleted_at = datetime.utcnow()
            self.db.commit()
        else:
            # Hard delete
            self.db.delete(db_obj)
            self.db.commit()
        
        return True
    
    def restore(self, id: UUID) -> bool:
        """
        Restore soft-deleted record.
        
        Args:
            id: Record UUID
            
        Returns:
            True if successful, False if not found or not deleted
        """
        if not hasattr(self.model, 'deleted_at'):
            return False
        
        db_obj = self.get(id, include_deleted=True)
        if not db_obj or not db_obj.deleted_at:
            return False
        
        db_obj.deleted_at = None
        self.db.commit()
        return True
    
    def exists(self, id: UUID, include_deleted: bool = False) -> bool:
        """
        Check if record exists.
        
        Args:
            id: Record UUID
            include_deleted: Whether to include soft-deleted records
            
        Returns:
            True if record exists, False otherwise
        """
        query = self.db.query(self.model.id).filter(self.model.id == id)
        
        if hasattr(self.model, 'deleted_at') and not include_deleted:
            query = query.filter(self.model.deleted_at.is_(None))
        
        return self.db.query(query.exists()).scalar()
    
    def count(
        self,
        include_deleted: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count records.
        
        Args:
            include_deleted: Whether to include soft-deleted records
            filters: Dictionary of column:value filters
            
        Returns:
            Count of records
        """
        query = self.db.query(func.count(self.model.id))
        
        if hasattr(self.model, 'deleted_at') and not include_deleted:
            query = query.filter(self.model.deleted_at.is_(None))
        
        if filters:
            for column, value in filters.items():
                if hasattr(self.model, column):
                    query = query.filter(getattr(self.model, column) == value)
        
        return query.scalar() or 0
    
    def bulk_create(self, objects: List[Dict[str, Any]]) -> List[ModelType]:
        """
        Bulk create records.
        
        Args:
            objects: List of dictionaries with object data
            
        Returns:
            List of created model instances
        """
        db_objs = [self.model(**obj_data) for obj_data in objects]
        self.db.bulk_save_objects(db_objs, return_defaults=True)
        self.db.commit()
        return db_objs
    
    def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """
        Bulk update records.
        
        Args:
            updates: List of dicts with 'id' and fields to update
            
        Returns:
            Number of records updated
        """
        count = 0
        for update_data in updates:
            if 'id' not in update_data:
                continue
            
            record_id = update_data.pop('id')
            if self.update(record_id, update_data):
                count += 1
        
        return count
    
    def bulk_delete(
        self,
        ids: List[UUID],
        soft: bool = True
    ) -> Tuple[int, List[UUID]]:
        """
        Bulk delete records.
        
        Args:
            ids: List of record UUIDs
            soft: Whether to soft delete
            
        Returns:
            Tuple of (success count, list of failed IDs)
        """
        success_count = 0
        failed_ids = []
        
        for record_id in ids:
            if self.delete(record_id, soft=soft):
                success_count += 1
            else:
                failed_ids.append(record_id)
        
        return success_count, failed_ids
    
    def search(
        self,
        search_term: str,
        search_fields: List[str],
        skip: int = 0,
        limit: int = 100,
        include_deleted: bool = False
    ) -> Tuple[List[ModelType], int]:
        """
        Search records across multiple fields.
        
        Args:
            search_term: Term to search for
            search_fields: List of field names to search in
            skip: Number of records to skip
            limit: Maximum number of records to return
            include_deleted: Whether to include soft-deleted records
            
        Returns:
            Tuple of (list of records, total count)
        """
        query = self.db.query(self.model)
        
        # Soft delete filter
        if hasattr(self.model, 'deleted_at') and not include_deleted:
            query = query.filter(self.model.deleted_at.is_(None))
        
        # Build search filter
        search_pattern = f'%{search_term}%'
        search_conditions = []
        
        for field in search_fields:
            if hasattr(self.model, field):
                column = getattr(self.model, field)
                search_conditions.append(column.ilike(search_pattern))
        
        if search_conditions:
            query = query.filter(or_(*search_conditions))
        
        # Get total
        total = query.count()
        
        # Apply pagination
        records = query.offset(skip).limit(limit).all()
        
        return records, total
    
    def get_by_field(
        self,
        field_name: str,
        field_value: Any,
        include_deleted: bool = False
    ) -> Optional[ModelType]:
        """
        Get record by specific field value.
        
        Args:
            field_name: Name of the field to filter by
            field_value: Value to match
            include_deleted: Whether to include soft-deleted records
            
        Returns:
            Model instance or None if not found
        """
        if not hasattr(self.model, field_name):
            return None
        
        query = self.db.query(self.model).filter(
            getattr(self.model, field_name) == field_value
        )
        
        if hasattr(self.model, 'deleted_at') and not include_deleted:
            query = query.filter(self.model.deleted_at.is_(None))
        
        return query.first()
    
    def get_multi_by_field(
        self,
        field_name: str,
        field_value: Any,
        skip: int = 0,
        limit: int = 100,
        include_deleted: bool = False
    ) -> Tuple[List[ModelType], int]:
        """
        Get multiple records by specific field value.
        
        Args:
            field_name: Name of the field to filter by
            field_value: Value to match
            skip: Number of records to skip
            limit: Maximum number of records to return
            include_deleted: Whether to include soft-deleted records
            
        Returns:
            Tuple of (list of records, total count)
        """
        if not hasattr(self.model, field_name):
            return [], 0
        
        query = self.db.query(self.model).filter(
            getattr(self.model, field_name) == field_value
        )
        
        if hasattr(self.model, 'deleted_at') and not include_deleted:
            query = query.filter(self.model.deleted_at.is_(None))
        
        total = query.count()
        
        # Order by created_at if available
        if hasattr(self.model, 'created_at'):
            query = query.order_by(desc(self.model.created_at))
        
        records = query.offset(skip).limit(limit).all()
        
        return records, total
    
    def refresh(self, db_obj: ModelType) -> ModelType:
        """
        Refresh model instance from database.
        
        Args:
            db_obj: Model instance to refresh
            
        Returns:
            Refreshed model instance
        """
        self.db.refresh(db_obj)
        return db_obj
    
    def commit(self) -> None:
        """Commit current transaction."""
        self.db.commit()
    
    def rollback(self) -> None:
        """Rollback current transaction."""
        self.db.rollback()
    
    def flush(self) -> None:
        """Flush changes to database without committing."""
        self.db.flush()