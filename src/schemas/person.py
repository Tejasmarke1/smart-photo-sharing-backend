"""Person schemas for face clustering and labeling."""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime
from uuid import UUID


# Base schemas
class PersonBase(BaseModel):
    """Base person schema."""
    name: Optional[str] = Field(None, max_length=255, description="Person's name")
    phone: Optional[str] = Field(None, max_length=20, description="Phone number")
    email: Optional[EmailStr] = Field(None, description="Email address")
    extra_data: Optional[str] = Field(None, description="Additional metadata (JSON)")


class PersonCreate(PersonBase):
    """Schema for creating a person."""
    album_id: UUID = Field(..., description="Album ID")
    representative_face_id: Optional[UUID] = Field(None, description="Representative face ID")


class PersonUpdate(BaseModel):
    """Schema for updating a person."""
    name: Optional[str] = Field(None, max_length=255)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[EmailStr] = None
    representative_face_id: Optional[UUID] = None
    extra_data: Optional[str] = None


class PersonInDB(PersonBase):
    """Person schema with database fields."""
    id: UUID
    album_id: UUID
    representative_face_id: Optional[UUID]
    face_count: int = Field(0, description="Number of faces assigned to this person")
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PersonWithFaces(PersonInDB):
    """Person with face details."""
    faces: List['FaceInDB'] = Field(default_factory=list, description="List of faces")


class PersonWithPhotos(PersonInDB):
    """Person with photo details."""
    photo_ids: List[UUID] = Field(default_factory=list, description="List of photo IDs")
    photo_count: int = Field(0, description="Number of unique photos")


class PersonSummary(BaseModel):
    """Minimal person info for lists."""
    id: UUID
    name: Optional[str]
    face_count: int
    representative_face_id: Optional[UUID]
    thumbnail_url: Optional[str] = None

    class Config:
        from_attributes = True


# Merge operations
class PersonMergeRequest(BaseModel):
    """Request to merge two persons."""
    source_person_id: UUID = Field(..., description="Person to merge from (will be deleted)")
    target_person_id: UUID = Field(..., description="Person to merge into (will be kept)")
    keep_source_name: bool = Field(False, description="Use source person's name")


class PersonMergeResponse(BaseModel):
    """Response after merging persons."""
    merged_person: PersonInDB
    faces_transferred: int
    source_person_deleted: bool


# Batch operations
class BatchLabelRequest(BaseModel):
    """Request to label multiple faces."""
    face_ids: List[UUID] = Field(..., min_length=1, description="Face IDs to label")
    person_id: Optional[UUID] = Field(None, description="Existing person ID, or None to create new")
    person_name: Optional[str] = Field(None, description="Name for new person")
    album_id: Optional[UUID] = Field(None, description="Album ID for new person")


class BatchLabelResponse(BaseModel):
    """Response after batch labeling."""
    person: PersonInDB
    faces_labeled: int
    created_new_person: bool


class BatchMergeRequest(BaseModel):
    """Request to merge multiple persons."""
    person_ids: List[UUID] = Field(..., min_length=2, description="Persons to merge")
    target_person_id: Optional[UUID] = Field(None, description="Target person, or None to use first")
    merged_name: Optional[str] = Field(None, description="Name for merged person")


class BatchMergeResponse(BaseModel):
    """Response after batch merge."""
    merged_person: PersonInDB
    persons_merged: int
    total_faces: int


# Split operations
class PersonSplitRequest(BaseModel):
    """Request to split a person into two."""
    person_id: UUID = Field(..., description="Person to split")
    face_ids_to_split: List[UUID] = Field(..., min_length=1, description="Faces to move to new person")
    new_person_name: Optional[str] = Field(None, description="Name for new person")


class PersonSplitResponse(BaseModel):
    """Response after splitting person."""
    original_person: PersonInDB
    new_person: PersonInDB
    faces_moved: int


# Transfer operations
class PersonTransferRequest(BaseModel):
    """Request to transfer person between albums."""
    person_id: UUID = Field(..., description="Person to transfer")
    target_album_id: UUID = Field(..., description="Target album ID")
    merge_if_exists: bool = Field(False, description="Merge with existing person if name matches")


class PersonTransferResponse(BaseModel):
    """Response after transfer."""
    person: PersonInDB
    faces_transferred: int
    merged_with_existing: bool
    merged_person_id: Optional[UUID] = None


# Similar persons
class SimilarPerson(BaseModel):
    """Similar person found."""
    person_id: UUID
    name: Optional[str]
    face_count: int
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    common_photos: int = Field(0, description="Number of common photos")
    thumbnail_url: Optional[str] = None


class SimilarPersonsResponse(BaseModel):
    """Response with similar persons."""
    person_id: UUID
    similar_persons: List[SimilarPerson]
    total_found: int


# List responses
class PersonListResponse(BaseModel):
    """Paginated person list."""
    persons: List[PersonInDB]
    total: int
    page: int
    page_size: int
    has_more: bool


# Thumbnail response
class PersonThumbnailResponse(BaseModel):
    """Person thumbnail info."""
    person_id: UUID
    thumbnail_url: Optional[str]
    face_id: Optional[UUID]
    s3_key: Optional[str]


# Import for forward references
from src.schemas.face import FaceInDB  # noqa: E402

PersonWithFaces.model_rebuild()
