"""Firebase Cloud Messaging service for push notifications."""
import logging
import json
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# Lazy-init flag
_firebase_initialized = False


def _ensure_firebase():
    """Initialize Firebase Admin SDK once."""
    global _firebase_initialized
    if _firebase_initialized:
        return
    try:
        import firebase_admin
        from firebase_admin import credentials
        from src.app.config import settings
        
        cred_path = getattr(settings, 'FIREBASE_SERVICE_ACCOUNT_PATH', 'firebase-service-account.json')
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        logger.info("Firebase Admin SDK initialized successfully")
    except FileNotFoundError:
        logger.warning("Firebase service account file not found. Push notifications will be simulated.")
    except Exception as e:
        logger.warning(f"Firebase init failed: {e}. Push notifications will be simulated.")


class FCMService:
    """Send push notifications via Firebase Cloud Messaging."""
    
    @staticmethod
    def send_to_tokens(
        tokens: List[str],
        title: str,
        body: str,
        image_url: Optional[str] = None,
        data: Optional[Dict[str, str]] = None
    ) -> Tuple[int, int, List[str]]:
        """
        Send notification to a list of FCM tokens.
        
        Returns:
            Tuple of (success_count, failure_count, failed_tokens)
        """
        if not tokens:
            return 0, 0, []
        
        _ensure_firebase()
        
        if not _firebase_initialized:
            # Simulate sending in dev mode
            logger.info(f"[SIMULATED] Push notification to {len(tokens)} devices: {title}")
            return len(tokens), 0, []
        
        try:
            from firebase_admin import messaging
            
            notification = messaging.Notification(
                title=title,
                body=body,
                image=image_url,
            )
            
            total_success = 0
            total_failure = 0
            failed_tokens = []
            
            # FCM supports max 500 tokens per multicast
            batch_size = 500
            for i in range(0, len(tokens), batch_size):
                batch = tokens[i:i + batch_size]
                message = messaging.MulticastMessage(
                    tokens=batch,
                    notification=notification,
                    data=data or {},
                    android=messaging.AndroidConfig(
                        priority='high',
                        notification=messaging.AndroidNotification(
                            channel_id='lumina_notifications',
                            icon='ic_notification',
                        )
                    ),
                    apns=messaging.APNSConfig(
                        payload=messaging.APNSPayload(
                            aps=messaging.Aps(
                                mutable_content=True,
                                sound='default',
                                badge=1,
                            )
                        )
                    ),
                )
                
                response = messaging.send_each_for_multicast(message)
                total_success += response.success_count
                total_failure += response.failure_count
                
                # Collect failed tokens for cleanup
                for idx, send_response in enumerate(response.responses):
                    if not send_response.success:
                        failed_tokens.append(batch[idx])
                        if send_response.exception:
                            logger.warning(f"FCM send failed for token: {send_response.exception}")
            
            logger.info(f"FCM batch send: {total_success} success, {total_failure} failures")
            return total_success, total_failure, failed_tokens
            
        except Exception as e:
            logger.error(f"FCM send_to_tokens failed: {e}")
            return 0, len(tokens), tokens
