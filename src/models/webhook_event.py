from sqlalchemy import Column, Integer, String, DateTime, JSON , Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.db.base import Base
import uuid
from .base import TimestampMixin
from datetime import datetime

class WebhookEvent(Base, TimestampMixin):
    """Webhook Event model to log incoming webhook events."""
    __tablename__ = 'webhook_events'
    
    id =Column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4,index=True)
    provider = Column(String(100), nullable=False, index=True)
    event_id = Column(String(255), nullable=False, unique=True, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    
    
    resource_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    resource_type = Column(String(100), nullable=True, index=True)
    
    
    payload = Column(JSON, nullable=False)
    
    processed = Column(Boolean, default=False, nullable=False, index=True)
    processed_at = Column(DateTime, nullable=True)
    received_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    
    def __repr__(self) -> str:
        return f'<WebhookEvent(id={self.id}, provider={self.provider}, event_type={self.event_type}, processed={self.processed})>'
    
    
    
    
    