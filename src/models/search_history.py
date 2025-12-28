"""Search history model for analytics and feedback."""
from sqlalchemy import Column, String, Integer, Float, ForeignKey, Boolean, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.db.base import Base
from .base import TimestampMixin


class SearchHistory(Base, TimestampMixin):
    """
    Track user search queries for analytics and improvement.
    
    Supports:
    - Search history tracking
    - Feedback collection
    - Performance analytics
    - ML model improvement
    """
    
    __tablename__ = 'search_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Search type and parameters
    search_type = Column(String(50), nullable=False, index=True)  # multi-face, by-person, cross-album, etc.
    query_params = Column(JSON, nullable=False)  # Original search parameters
    
    # Results metadata
    result_count = Column(Integer, nullable=False, default=0)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Albums/faces involved
    album_ids = Column(JSON, nullable=True)  # Array of album IDs searched
    query_face_ids = Column(JSON, nullable=True)  # Face IDs used in query
    
    # Feedback
    feedback_given = Column(Boolean, nullable=False, default=False)
    relevant_face_ids = Column(JSON, nullable=True)  # User-marked relevant results
    irrelevant_face_ids = Column(JSON, nullable=True)  # User-marked irrelevant results
    feedback_comments = Column(Text, nullable=True)
    
    # Quality metrics
    avg_similarity = Column(Float, nullable=True)
    avg_relevance_score = Column(Float, nullable=True)  # If feedback provided
    
    # Flags for analytics
    missing_expected = Column(Boolean, nullable=False, default=False)
    too_many_results = Column(Boolean, nullable=False, default=False)
    
    # Additional metadata
    extra_data = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship('User', backref='search_history')
    
    def __repr__(self) -> str:
        return f'<SearchHistory(id={self.id}, type={self.search_type}, user={self.user_id})>'
