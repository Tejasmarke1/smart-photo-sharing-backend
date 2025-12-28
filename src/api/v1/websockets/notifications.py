"""
Production-Grade WebSocket Notification System
==============================================

Real-time notification system using WebSockets for instant updates.
Replaces HTTP webhooks with persistent WebSocket connections.

Features:
- Real-time bidirectional communication
- Connection management and heartbeat
- User-specific notification channels
- Event-based notification routing
- Automatic reconnection support
- Message queuing for offline users
- Broadcast and targeted messaging
- Comprehensive error handling
- Connection statistics and monitoring
"""

import logging
import json
import asyncio
from typing import Dict, Set, List, Optional, Any
from uuid import UUID
from datetime import datetime
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect, Depends, status, APIRouter, Query
from fastapi.websockets import WebSocketState
from sqlalchemy.orm import Session

from src.models.user import User
from src.api.deps import get_current_user_ws
from src.db.base import get_db
from src.core.security import decode_token


logger = logging.getLogger(__name__)

# Create router for WebSocket endpoints
router = APIRouter()


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """
    Manages WebSocket connections for real-time notifications.
    
    Features:
    - User-specific connections
    - Connection pooling
    - Message broadcasting
    - Offline message queuing
    """
    
    def __init__(self):
        """Initialize connection manager."""
        # Active connections: user_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        
        # Message queue for offline users: user_id -> list of messages
        self.message_queue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Connection metadata: websocket -> user info
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Statistics
        self._stats = {
            "total_connections": 0,
            "messages_sent": 0,
            "messages_queued": 0,
            "disconnections": 0,
            "errors": 0
        }
        
        logger.info("🔌 WebSocket Connection Manager initialized")
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: UUID,
        user_name: Optional[str] = None
    ):
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            user_id: User ID
            user_name: Optional user name for logging
        """
        await websocket.accept()
        
        user_id_str = str(user_id)
        self.active_connections[user_id_str].add(websocket)
        
        # Store connection metadata
        self.connection_info[websocket] = {
            "user_id": user_id_str,
            "user_name": user_name,
            "connected_at": datetime.utcnow(),
            "messages_sent": 0
        }
        
        self._stats["total_connections"] += 1
        
        logger.info(
            f"✅ User {user_name or user_id_str} connected "
            f"(total connections: {self.get_total_connections()})"
        )
        
        # Send queued messages if any
        await self._send_queued_messages(websocket, user_id_str)
    
    async def disconnect(self, websocket: WebSocket):
        """
        Disconnect and cleanup a WebSocket connection.
        
        Args:
            websocket: WebSocket connection to disconnect
        """
        if websocket in self.connection_info:
            info = self.connection_info[websocket]
            user_id = info["user_id"]
            user_name = info.get("user_name", user_id)
            
            # Remove from active connections
            if user_id in self.active_connections:
                self.active_connections[user_id].discard(websocket)
                
                # Clean up if no more connections for this user
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            
            # Remove metadata
            del self.connection_info[websocket]
            
            self._stats["disconnections"] += 1
            
            logger.info(
                f"❌ User {user_name} disconnected "
                f"(total connections: {self.get_total_connections()})"
            )
    
    async def send_personal_message(
        self,
        message: Dict[str, Any],
        user_id: UUID
    ):
        """
        Send message to specific user.
        
        Args:
            message: Message dictionary to send
            user_id: Target user ID
        """
        user_id_str = str(user_id)
        
        # Check if user has active connections
        if user_id_str in self.active_connections:
            connections = self.active_connections[user_id_str].copy()
            
            for websocket in connections:
                try:
                    await websocket.send_json(message)
                    
                    # Update stats
                    if websocket in self.connection_info:
                        self.connection_info[websocket]["messages_sent"] += 1
                    self._stats["messages_sent"] += 1
                    
                    logger.debug(f"📤 Sent message to user {user_id_str}: {message.get('event')}")
                    
                except WebSocketDisconnect:
                    await self.disconnect(websocket)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id_str}: {str(e)}")
                    self._stats["errors"] += 1
                    await self.disconnect(websocket)
        else:
            # Queue message for offline user
            self.message_queue[user_id_str].append({
                **message,
                "queued_at": datetime.utcnow().isoformat()
            })
            self._stats["messages_queued"] += 1
            
            # Limit queue size per user
            if len(self.message_queue[user_id_str]) > 100:
                self.message_queue[user_id_str] = self.message_queue[user_id_str][-100:]
            
            logger.debug(f"📥 Queued message for offline user {user_id_str}")
    
    async def send_to_multiple_users(
        self,
        message: Dict[str, Any],
        user_ids: List[UUID]
    ):
        """
        Send message to multiple specific users.
        
        Args:
            message: Message to send
            user_ids: List of target user IDs
        """
        tasks = [
            self.send_personal_message(message, user_id)
            for user_id in user_ids
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude_user: Optional[UUID] = None
    ):
        """
        Broadcast message to all connected users.
        
        Args:
            message: Message to broadcast
            exclude_user: Optional user ID to exclude from broadcast
        """
        exclude_id = str(exclude_user) if exclude_user else None
        broadcast_count = 0
        
        for user_id, connections in list(self.active_connections.items()):
            if exclude_id and user_id == exclude_id:
                continue
            
            for websocket in list(connections):
                try:
                    await websocket.send_json(message)
                    
                    if websocket in self.connection_info:
                        self.connection_info[websocket]["messages_sent"] += 1
                    
                    broadcast_count += 1
                    self._stats["messages_sent"] += 1
                    
                except WebSocketDisconnect:
                    await self.disconnect(websocket)
                except Exception as e:
                    logger.error(f"Error broadcasting to user {user_id}: {str(e)}")
                    self._stats["errors"] += 1
                    await self.disconnect(websocket)
        
        logger.info(f"📢 Broadcast message to {broadcast_count} connections: {message.get('event')}")
    
    async def _send_queued_messages(self, websocket: WebSocket, user_id: str):
        """Send all queued messages to newly connected user."""
        if user_id in self.message_queue and self.message_queue[user_id]:
            queued = self.message_queue[user_id]
            logger.info(f"📬 Sending {len(queued)} queued messages to user {user_id}")
            
            for message in queued:
                try:
                    await websocket.send_json(message)
                    self._stats["messages_sent"] += 1
                except Exception as e:
                    logger.error(f"Error sending queued message: {str(e)}")
                    break
            
            # Clear queue
            del self.message_queue[user_id]
    
    def get_user_connections(self, user_id: UUID) -> int:
        """Get number of active connections for a user."""
        return len(self.active_connections.get(str(user_id), set()))
    
    def get_total_connections(self) -> int:
        """Get total number of active connections."""
        return sum(len(conns) for conns in self.active_connections.values())
    
    def get_connected_users(self) -> List[str]:
        """Get list of connected user IDs."""
        return list(self.active_connections.keys())
    
    def is_user_online(self, user_id: UUID) -> bool:
        """Check if user has any active connections."""
        return str(user_id) in self.active_connections
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            **self._stats,
            "active_connections": self.get_total_connections(),
            "connected_users": len(self.active_connections),
            "queued_messages": sum(len(q) for q in self.message_queue.values())
        }


# =============================================================================
# Global Connection Manager Instance
# =============================================================================

manager = ConnectionManager()


# =============================================================================
# Notification Helper Functions
# =============================================================================

async def notify_user(
    user_id: UUID,
    event: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Send notification to specific user.
    
    Args:
        user_id: Target user ID
        event: Event type (e.g., "person.created")
        data: Event data
        metadata: Optional metadata
    
    Example:
        ```python
        await notify_user(
            user_id=current_user.id,
            event="person.created",
            data={
                "person_id": str(person.id),
                "name": person.name,
                "album_id": str(album.id)
            }
        )
        ```
    """
    message = {
        "event": event,
        "data": data,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await manager.send_personal_message(message, user_id)


async def notify_multiple_users(
    user_ids: List[UUID],
    event: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Send notification to multiple users.
    
    Args:
        user_ids: List of target user IDs
        event: Event type
        data: Event data
        metadata: Optional metadata
    """
    message = {
        "event": event,
        "data": data,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await manager.send_to_multiple_users(message, user_ids)


async def broadcast_notification(
    event: str,
    data: Dict[str, Any],
    exclude_user: Optional[UUID] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Broadcast notification to all connected users.
    
    Args:
        event: Event type
        data: Event data
        exclude_user: Optional user to exclude
        metadata: Optional metadata
    """
    message = {
        "event": event,
        "data": data,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await manager.broadcast(message, exclude_user)


# =============================================================================
# Convenience Functions (Webhook Replacement)
# =============================================================================

async def trigger_notification(
    event: str,
    data: Dict[str, Any],
    user_id: Optional[UUID] = None,
    user_ids: Optional[List[UUID]] = None,
    broadcast: bool = False,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Main function to trigger notifications (replaces trigger_webhook).
    
    This is a drop-in replacement for the webhook trigger_webhook function.
    
    Args:
        event: Event type (e.g., "person.created", "photo.uploaded")
        data: Event data
        user_id: Single user to notify
        user_ids: Multiple users to notify
        broadcast: If True, broadcast to all users
        metadata: Optional metadata
    
    Example:
        ```python
        # Single user notification
        background_tasks.add_task(
            trigger_notification,
            event="person.created",
            data={"person_id": str(person.id), "name": person.name},
            user_id=current_user.id
        )
        
        # Broadcast notification
        background_tasks.add_task(
            trigger_notification,
            event="system.maintenance",
            data={"message": "System maintenance in 10 minutes"},
            broadcast=True
        )
        ```
    """
    try:
        if broadcast:
            await broadcast_notification(event, data, metadata=metadata)
        elif user_ids:
            await notify_multiple_users(user_ids, event, data, metadata)
        elif user_id:
            await notify_user(user_id, event, data, metadata)
        else:
            logger.warning(f"No target specified for notification: {event}")
        
        logger.debug(f"✅ Triggered notification: {event}")
        
    except Exception as e:
        logger.error(f"Failed to trigger notification: {str(e)}", exc_info=True)


# Alias for backward compatibility with webhook code
trigger_webhook = trigger_notification


def get_connection_stats() -> Dict[str, Any]:
    """Get WebSocket connection statistics."""
    return manager.get_statistics()


def is_user_online(user_id: UUID) -> bool:
    """Check if user is currently connected."""
    return manager.is_user_online(user_id)


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@router.get("/ws/stats")
async def get_websocket_stats():
    """
    Get WebSocket connection statistics.
    
    Returns statistics about active connections, messages sent, etc.
    """
    return get_connection_stats()


@router.websocket("/ws/notifications")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT access token"),
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time notifications.
    
    Connection URL: ws://localhost:8000/api/v1/ws/notifications?token=YOUR_JWT_TOKEN
    
    Features:
    - Real-time bidirectional communication
    - Automatic reconnection support
    - Message queuing for offline periods
    - Heartbeat/ping-pong for connection health
    
    Authentication:
    - Pass JWT token as query parameter: ?token=YOUR_ACCESS_TOKEN
    - Connection will be rejected if token is invalid
    
    Message Format (Server -> Client):
    {
        "event": "person.created",
        "data": {
            "person_id": "uuid",
            "name": "John Doe",
            "album_id": "uuid"
        },
        "metadata": {},
        "timestamp": "2025-01-01T00:00:00Z"
    }
    
    Client Messages (Client -> Server):
    {
        "type": "ping"  // Heartbeat
    }
    or
    {
        "type": "subscribe",
        "events": ["person.created", "photo.uploaded"]  // Optional event filtering
    }
    
    Example JavaScript Client:
    ```javascript
    const token = "your-jwt-token";
    const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/notifications?token=${token}`);
    
    ws.onopen = () => {
        console.log('Connected to notifications');
        // Send heartbeat every 30 seconds
        setInterval(() => {
            ws.send(JSON.stringify({type: 'ping'}));
        }, 30000);
    };
    
    ws.onmessage = (event) => {
        const notification = JSON.parse(event.data);
        console.log('Received:', notification);
        
        // Handle different events
        switch(notification.event) {
            case 'person.created':
                showNotification('New person!', notification.data);
                break;
            case 'photo.processed':
                updateGallery(notification.data);
                break;
        }
    };
    
    ws.onclose = () => {
        console.log('Disconnected, reconnecting...');
        setTimeout(() => location.reload(), 1000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    ```
    """
    
    # Authenticate user
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing token")
        logger.warning("WebSocket connection rejected: Missing token")
        return
    
    # Decode and verify token
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        logger.warning("WebSocket connection rejected: Invalid token")
        return
    
    user_id = payload.get("sub")
    if not user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token payload")
        logger.warning("WebSocket connection rejected: Invalid token payload")
        return
    
    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="User not found or inactive")
        logger.warning(f"WebSocket connection rejected: User {user_id} not found or inactive")
        return
    
    # Accept connection and register with manager
    await manager.connect(websocket, user.id, user.email)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "event": "connection.established",
            "data": {
                "user_id": str(user.id),
                "user_email": user.email,
                "connected_at": datetime.utcnow().isoformat(),
                "message": "Successfully connected to notification service"
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60.0  # 60 second timeout
                )
                
                # Parse client message
                try:
                    message = json.loads(data)
                    message_type = message.get("type")
                    
                    if message_type == "ping":
                        # Respond to heartbeat
                        await websocket.send_json({
                            "event": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        logger.debug(f"Heartbeat from user {user.email}")
                    
                    elif message_type == "subscribe":
                        # Handle event subscription (future feature)
                        events = message.get("events", [])
                        await websocket.send_json({
                            "event": "subscription.updated",
                            "data": {"subscribed_events": events},
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        logger.info(f"User {user.email} subscribed to events: {events}")
                    
                    elif message_type == "status":
                        # Send connection status
                        stats = manager.get_statistics()
                        await websocket.send_json({
                            "event": "status.response",
                            "data": {
                                "user_connections": manager.get_user_connections(user.id),
                                "total_connections": stats["active_connections"],
                                "messages_sent": manager.connection_info[websocket]["messages_sent"]
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    else:
                        logger.warning(f"Unknown message type from user {user.email}: {message_type}")
                
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from user {user.email}: {data}")
                    await websocket.send_json({
                        "event": "error",
                        "data": {"message": "Invalid JSON format"},
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({
                        "event": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except:
                    # Connection lost
                    break
            
            except WebSocketDisconnect:
                break
            
            except Exception as e:
                logger.error(f"Error handling message from user {user.email}: {str(e)}", exc_info=True)
                break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {user.email}")
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user.email}: {str(e)}", exc_info=True)
    
    finally:
        # Clean up connection
        await manager.disconnect(websocket)
        logger.info(f"WebSocket connection closed for user {user.email}")
