"""Person repository for database operations."""
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, and_, or_, case
import logging

from src.models.person import Person
from src.models.face_person import FacePerson
from src.models.face import Face
from src.models.photo import Photo
from src.schemas.person import PersonCreate, PersonUpdate

logger = logging.getLogger(__name__)


class PersonRepository:
    """Repository for person operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, person_data: PersonCreate) -> Person:
        """Create a new person."""
        person = Person(
            album_id=person_data.album_id,
            name=person_data.name,
            phone=person_data.phone,
            email=person_data.email,
            representative_face_id=person_data.representative_face_id,
            extra_data=person_data.extra_data
        )
        self.db.add(person)
        self.db.flush()
        return person
    
    def get_by_id(self, person_id: UUID) -> Optional[Person]:
        """Get person by ID."""
        return self.db.query(Person).filter(
            Person.id == person_id,
            Person.deleted_at.is_(None)
        ).first()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        album_id: Optional[UUID] = None
    ) -> Tuple[List[Person], int]:
        """Get all persons with pagination."""
        query = self.db.query(Person).filter(Person.deleted_at.is_(None))
        
        if album_id:
            query = query.filter(Person.album_id == album_id)
        
        total = query.count()
        persons = query.order_by(Person.created_at.desc()).offset(skip).limit(limit).all()
        
        return persons, total
    
    def get_by_album(self, album_id: UUID) -> List[Person]:
        """Get all persons in an album."""
        return self.db.query(Person).filter(
            Person.album_id == album_id,
            Person.deleted_at.is_(None)
        ).all()
    
    def update(self, person_id: UUID, person_data: PersonUpdate) -> Optional[Person]:
        """Update person details."""
        person = self.get_by_id(person_id)
        if not person:
            return None
        
        update_data = person_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(person, field, value)
        
        self.db.flush()
        return person
    
    def delete(self, person_id: UUID) -> bool:
        """Soft delete a person."""
        person = self.get_by_id(person_id)
        if not person:
            return False
        
        person.soft_delete()
        self.db.flush()
        return True
    
    def hard_delete(self, person_id: UUID) -> bool:
        """Hard delete a person (use with caution)."""
        person = self.get_by_id(person_id)
        if not person:
            return False
        
        self.db.delete(person)
        self.db.flush()
        return True
    
    def get_faces(self, person_id: UUID) -> List[Face]:
        """Get all faces assigned to a person."""
        return (
            self.db.query(Face)
            .join(FacePerson)
            .filter(
                FacePerson.person_id == person_id,
                Face.deleted_at.is_(None)
            )
            .all()
        )
    
    def get_photos(self, person_id: UUID) -> List[UUID]:
        """Get all unique photo IDs containing this person."""
        photo_ids = (
            self.db.query(Face.photo_id)
            .join(FacePerson)
            .filter(
                FacePerson.person_id == person_id,
                Face.deleted_at.is_(None)
            )
            .distinct()
            .all()
        )
        return [photo_id[0] for photo_id in photo_ids]
    
    def get_thumbnail(self, person_id: UUID) -> Optional[Face]:
        """Get best representative face for thumbnail."""
        person = self.get_by_id(person_id)
        if not person:
            return None
        
        # Use representative face if set
        if person.representative_face_id:
            face = self.db.query(Face).filter(
                Face.id == person.representative_face_id,
                Face.deleted_at.is_(None)
            ).first()
            if face:
                return face
        
        # Otherwise, get best quality face (highest confidence, good blur/brightness)
        face = (
            self.db.query(Face)
            .join(FacePerson)
            .filter(
                FacePerson.person_id == person_id,
                Face.deleted_at.is_(None),
                Face.thumbnail_s3_key.isnot(None)
            )
            .order_by(
                Face.confidence.desc(),
                Face.blur_score.desc(),
                Face.brightness_score.desc()
            )
            .first()
        )
        
        return face
    
    def merge_persons(
        self,
        source_person_id: UUID,
        target_person_id: UUID,
        keep_source_name: bool = False
    ) -> Optional[Person]:
        """Merge two persons."""
        source = self.get_by_id(source_person_id)
        target = self.get_by_id(target_person_id)
        
        if not source or not target:
            return None
        
        # Transfer all face mappings from source to target
        self.db.query(FacePerson).filter(
            FacePerson.person_id == source_person_id
        ).update({FacePerson.person_id: target_person_id})
        
        # Optionally keep source name
        if keep_source_name and source.name:
            target.name = source.name
        
        # Delete source person
        source.soft_delete()
        
        self.db.flush()
        return target
    
    def batch_label_faces(
        self,
        face_ids: List[UUID],
        person_id: UUID
    ) -> int:
        """Assign multiple faces to a person."""
        # Remove existing mappings for these faces
        self.db.query(FacePerson).filter(
            FacePerson.face_id.in_(face_ids)
        ).delete(synchronize_session=False)
        
        # Create new mappings
        mappings = [
            FacePerson(face_id=face_id, person_id=person_id)
            for face_id in face_ids
        ]
        self.db.bulk_save_objects(mappings)
        self.db.flush()
        
        return len(mappings)
    
    def batch_merge_persons(
        self,
        person_ids: List[UUID],
        target_person_id: UUID,
        merged_name: Optional[str] = None
    ) -> Optional[Person]:
        """Merge multiple persons into one."""
        if target_person_id not in person_ids:
            return None
        
        target = self.get_by_id(target_person_id)
        if not target:
            return None
        
        # Get all persons to merge
        source_ids = [pid for pid in person_ids if pid != target_person_id]
        
        # Transfer all face mappings to target
        self.db.query(FacePerson).filter(
            FacePerson.person_id.in_(source_ids)
        ).update({FacePerson.person_id: target_person_id}, synchronize_session=False)
        
        # Delete source persons
        self.db.query(Person).filter(
            Person.id.in_(source_ids)
        ).update({Person.deleted_at: func.now()}, synchronize_session=False)
        
        # Update target name if provided
        if merged_name:
            target.name = merged_name
        
        self.db.flush()
        return target
    
    def split_person(
        self,
        person_id: UUID,
        face_ids_to_split: List[UUID],
        new_person_name: Optional[str] = None
    ) -> Optional[Tuple[Person, Person]]:
        """Split a person into two."""
        original = self.get_by_id(person_id)
        if not original:
            return None
        
        # Create new person in same album
        new_person = Person(
            album_id=original.album_id,
            name=new_person_name or f"{original.name or 'Person'} (Split)"
        )
        self.db.add(new_person)
        self.db.flush()
        
        # Move specified faces to new person
        self.db.query(FacePerson).filter(
            FacePerson.person_id == person_id,
            FacePerson.face_id.in_(face_ids_to_split)
        ).update({FacePerson.person_id: new_person.id}, synchronize_session=False)
        
        self.db.flush()
        return original, new_person
    
    def transfer_to_album(
        self,
        person_id: UUID,
        target_album_id: UUID,
        merge_if_exists: bool = False
    ) -> Optional[Tuple[Person, bool, Optional[UUID]]]:
        """Transfer person to another album.
        
        Returns:
            Tuple of (person, was_merged, merged_with_person_id)
        """
        person = self.get_by_id(person_id)
        if not person:
            return None
        
        # Check if person with same name exists in target album
        existing = None
        if merge_if_exists and person.name:
            existing = self.db.query(Person).filter(
                Person.album_id == target_album_id,
                Person.name == person.name,
                Person.deleted_at.is_(None)
            ).first()
        
        if existing:
            # Merge into existing person
            self.merge_persons(person_id, existing.id)
            return existing, True, existing.id
        else:
            # Just change album
            person.album_id = target_album_id
            self.db.flush()
            return person, False, None
    
    def find_similar_persons(
        self,
        person_id: UUID,
        similarity_threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find persons with similar faces (potential duplicates).
        
        Uses face embeddings to compute similarity between persons.
        """
        person = self.get_by_id(person_id)
        if not person:
            return []
        
        # Get faces of this person
        person_faces = self.get_faces(person_id)
        if not person_faces:
            return []
        
        person_face_ids = [str(f.id) for f in person_faces]
        
        # Find other persons in same album
        other_persons = self.db.query(Person).filter(
            Person.album_id == person.album_id,
            Person.id != person_id,
            Person.deleted_at.is_(None)
        ).all()
        
        similar = []
        
        for other in other_persons:
            other_faces = self.get_faces(other.id)
            if not other_faces:
                continue
            
            # Compute average similarity between face sets
            # This is a placeholder - you'd need actual embedding comparison
            # For now, check for common photos as a proxy
            common_photos = set(self.get_photos(person_id)) & set(self.get_photos(other.id))
            
            if len(common_photos) > 0:
                # Rough similarity score based on common photos
                score = min(len(common_photos) / max(len(person_faces), len(other_faces)), 1.0)
                
                if score >= similarity_threshold:
                    similar.append({
                        'person_id': other.id,
                        'name': other.name,
                        'face_count': len(other_faces),
                        'similarity_score': score,
                        'common_photos': len(common_photos)
                    })
        
        # Sort by similarity score
        similar.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar[:limit]
    
    def get_face_count(self, person_id: UUID) -> int:
        """Get count of faces assigned to person."""
        return self.db.query(FacePerson).filter(
            FacePerson.person_id == person_id
        ).count()
    
    def exists(self, person_id: UUID) -> bool:
        """Check if person exists."""
        return self.db.query(Person).filter(
            Person.id == person_id,
            Person.deleted_at.is_(None)
        ).count() > 0
