"""
WebSocket-Based Real-Time Notification Service
==============================================

This module provides a WebSocket-based notification system that replaces
traditional HTTP webhooks with real-time bidirectional communication.

This is a compatibility layer that redirects to the WebSocket notification system.
For direct WebSocket usage, import from src.api.v1.websockets.notifications

Features:
- Real-time notifications via WebSockets
- Automatic connection management
- Message queuing for offline users
- Backward compatibility with webhook-style calls
- Much simpler and more efficient than HTTP webhooks
"""

import logging
from typing import Dict, Any, Optional, List, Union
from uuid import UUID

# Import the WebSocket notification system
from src.api.v1.websockets.notifications import (
    trigger_notification,
    notify_user,
    notify_multiple_users,
    broadcast_notification,
    get_connection_stats,
    is_user_online,
    manager as websocket_manager
)


logger = logging.getLogger(__name__)


# =============================================================================
# Backward Compatibility Layer
# =============================================================================

async def trigger_webhook(
    event: str,
    data: Dict[str, Any],
    user_id: Optional[UUID] = None,
    user_ids: Optional[List[UUID]] = None,
    broadcast: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Trigger real-time notification via WebSocket (replaces HTTP webhooks).
    
    This function provides backward compatibility with the old webhook system.
    It now sends real-time notifications through WebSocket connections.
    
    Args:
        event: Event type (e.g., "person.created", "photo.uploaded")
        data: Event data to send
        user_id: Optional single user to notify
        user_ids: Optional list of users to notify
        broadcast: If True, broadcast to all connected users
        metadata: Optional additional metadata
    
    Returns:
        Dictionary with delivery results
    
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
                trigger_webhook,
                event="resource.created",
                data={"resource_id": str(resource.id)},
                user_id=current_user.id
            )
        ```
    """
    try:
        await trigger_notification(
            event=event,
            data=data,
            user_id=user_id,
            user_ids=user_ids,
            broadcast=broadcast,
            metadata=metadata
        )
        
        # Return success response
        return {
            "status": "success",
            "event": event,
            "delivery_method": "websocket",
            "message": "Notification sent via WebSocket"
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger notification: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "event": event,
            "error": str(e),
            "delivery_method": "websocket"
        }


# =============================================================================
# WebSocket-Specific Functions
# =============================================================================

def get_webhook_stats() -> Dict[str, Any]:
    """
    Get WebSocket connection statistics.
    
    Returns:
        Dictionary with connection statistics
    """
    return get_connection_stats()


def check_user_online(user_id: UUID) -> bool:
    """
    Check if user is currently connected via WebSocket.
    
    Args:
        user_id: User ID to check
    
    Returns:
        True if user has active WebSocket connection
    """
    return is_user_online(user_id)


# =============================================================================
# Convenience Functions for Common Events
# =============================================================================

async def notify_person_created(
    user_id: UUID,
    person_id: UUID,
    person_name: str,
    album_id: UUID
):
    """Notify user about person creation."""
    await notify_user(
        user_id=user_id,
        event="person.created",
        data={
            "person_id": str(person_id),
            "name": person_name,
            "album_id": str(album_id)
        }
    )


async def notify_person_updated(
    user_id: UUID,
    person_id: UUID,
    person_name: str,
    changes: Dict[str, Any]
):
    """Notify user about person update."""
    await notify_user(
        user_id=user_id,
        event="person.updated",
        data={
            "person_id": str(person_id),
            "name": person_name,
            "changes": changes
        }
    )


async def notify_person_deleted(
    user_id: UUID,
    person_id: UUID,
    album_id: UUID
):
    """Notify user about person deletion."""
    await notify_user(
        user_id=user_id,
        event="person.deleted",
        data={
            "person_id": str(person_id),
            "album_id": str(album_id)
        }
    )


async def notify_person_merged(
    user_id: UUID,
    merged_person_id: UUID,
    source_person_id: UUID,
    target_person_id: UUID,
    faces_transferred: int
):
    """Notify user about person merge."""
    await notify_user(
        user_id=user_id,
        event="person.merged",
        data={
            "merged_person_id": str(merged_person_id),
            "source_person_id": str(source_person_id),
            "target_person_id": str(target_person_id),
            "faces_transferred": faces_transferred
        }
    )


async def notify_batch_completed(
    user_id: UUID,
    batch_type: str,
    total_items: int,
    successful: int,
    failed: int
):
    """Notify user about batch operation completion."""
    await notify_user(
        user_id=user_id,
        event="batch.completed",
        data={
            "batch_type": batch_type,
            "total_items": total_items,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful/total_items*100):.1f}%" if total_items > 0 else "0%"
        }
    )


async def notify_photo_processed(
    user_id: UUID,
    photo_id: UUID,
    album_id: UUID,
    faces_detected: int
):
    """Notify user about photo processing completion."""
    await notify_user(
        user_id=user_id,
        event="photo.processed",
        data={
            "photo_id": str(photo_id),
            "album_id": str(album_id),
            "faces_detected": faces_detected
        }
    )


async def notify_album_shared(
    user_ids: List[UUID],
    album_id: UUID,
    album_name: str,
    shared_by: str
):
    """Notify multiple users about album sharing."""
    await notify_multiple_users(
        user_ids=user_ids,
        event="album.shared",
        data={
            "album_id": str(album_id),
            "album_name": album_name,
            "shared_by": shared_by
        }
    )


async def broadcast_system_message(
    message: str,
    message_type: str = "info",
    action_required: bool = False
):
    """Broadcast system message to all users."""
    await broadcast_notification(
        event="system.message",
        data={
            "message": message,
            "type": message_type,
            "action_required": action_required
        }
    )


# =============================================================================
# Migration Notes
# =============================================================================

"""
MIGRATION FROM WEBHOOKS TO WEBSOCKETS
=====================================

This module now uses WebSockets instead of HTTP webhooks. Benefits:

✅ Real-time instant notifications (no HTTP polling or delays)
✅ Bidirectional communication (server can push to clients)
✅ Persistent connections (more efficient than HTTP)
✅ Automatic message queuing for offline users
✅ Simpler implementation (no retry logic, HMAC signatures, etc.)
✅ Lower latency and better user experience

BACKWARD COMPATIBILITY:
- All existing trigger_webhook() calls work without changes
- Function signatures remain the same
- Background tasks compatibility maintained

SETUP REQUIRED:
1. WebSocket endpoint must be registered in FastAPI app
2. Clients must connect to WebSocket endpoint
3. See src/api/v1/websockets/notifications.py for WebSocket router

EXAMPLE WEBSOCKET ENDPOINT:
```python
from fastapi import WebSocket, WebSocketDisconnect
from src.api.v1.websockets.notifications import manager
from src.api.deps import get_current_user_ws

@app.websocket("/ws/notifications")
async def websocket_endpoint(
    websocket: WebSocket,
    current_user: User = Depends(get_current_user_ws)
):
    await manager.connect(websocket, current_user.id, current_user.email)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
```

CLIENT-SIDE EXAMPLE (JavaScript):
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/notifications');

ws.onmessage = (event) => {
    const notification = JSON.parse(event.data);
    console.log('Received notification:', notification);
    
    // Handle different event types
    switch(notification.event) {
        case 'person.created':
            showNotification('New person created!', notification.data);
            break;
        case 'photo.processed':
            updatePhotoGallery(notification.data);
            break;
        // ... handle other events
    }
};

ws.onclose = () => {
    console.log('Disconnected, reconnecting...');
    setTimeout(() => location.reload(), 1000);
};
```
"""

