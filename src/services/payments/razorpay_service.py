"""Razorpay integration service."""
import logging
import razorpay
from typing import Optional, Dict, Any
from src.app.config import settings

logger = logging.getLogger(__name__)


class RazorpayService:
    """Service to handle Razorpay payment gateway operations."""

    def __init__(self):
        self.key_id = settings.RAZORPAY_KEY_ID
        self.key_secret = settings.RAZORPAY_KEY_SECRET
        self.webhook_secret = settings.RAZORPAY_WEBHOOK_SECRET
        
        # Initialize client. Handle empty credentials for development/mocking
        if self.key_id and self.key_secret:
            self.client = razorpay.Client(auth=(self.key_id, self.key_secret))
        else:
            self.client = None
            logger.warning("Razorpay credentials are not set in config.")

    def create_order(
        self, 
        amount_cents: int, 
        currency: str = "INR", 
        receipt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Razorpay order.
        
        Args:
            amount_cents: Amount in cents/paise (e.g. 50000 for Rs. 500)
            currency: Currency code (default: INR)
            receipt: Unique identifier for the transaction
            
        Returns:
            Dict containing the created order details
        """
        if not self.client:
            # Return dummy order data in development if credentials are empty
            dummy_id = f"order_dummy_{receipt or '123'}"
            logger.info(f"Dev mode: Creating dummy Razorpay order {dummy_id}")
            return {
                "id": dummy_id,
                "entity": "order",
                "amount": amount_cents,
                "amount_paid": 0,
                "amount_due": amount_cents,
                "currency": currency,
                "receipt": receipt,
                "status": "created",
                "attempts": 0,
                "notes": {},
                "created_at": 1600000000
            }

        try:
            data = {
                "amount": amount_cents,
                "currency": currency,
                "payment_capture": 1  # Auto capture payment
            }
            if receipt:
                data["receipt"] = receipt
                
            order = self.client.order.create(data=data)
            logger.info(f"Successfully created Razorpay order: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Error creating Razorpay order: {e}")
            raise

    def verify_payment_signature(
        self,
        razorpay_order_id: str,
        razorpay_payment_id: str,
        razorpay_signature: str
    ) -> bool:
        """
        Verify the signature of a payment returned by frontend.
        """
        if not self.client or razorpay_order_id.startswith("order_dummy_"):
            logger.info("Dev mode: Bypassing Razorpay signature verification")
            return True

        try:
            params_dict = {
                'razorpay_order_id': razorpay_order_id,
                'razorpay_payment_id': razorpay_payment_id,
                'razorpay_signature': razorpay_signature
            }
            self.client.utility.verify_payment_signature(params_dict)
            logger.info(f"Razorpay signature verified for order: {razorpay_order_id}")
            return True
        except Exception as e:
            logger.warning(f"Razorpay signature verification failed: {e}")
            return False

    def verify_webhook_signature(self, payload_body: bytes, signature: str) -> bool:
        """
        Verify the signature of an incoming webhook event.
        """
        if not self.client or not self.webhook_secret:
            logger.warning("Bypassing webhook signature verification: client or secret missing")
            return True

        try:
            # payload_body must be raw bytes/string
            self.client.utility.verify_webhook_signature(
                payload_body.decode('utf-8') if isinstance(payload_body, bytes) else payload_body,
                signature,
                self.webhook_secret
            )
            return True
        except Exception as e:
            logger.warning(f"Razorpay webhook signature verification failed: {e}")
            return False
