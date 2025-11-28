"""Send OTP via email/SMS/WhatsApp."""
import logging

logger = logging.getLogger(__name__)


class NotificationService:
    """Send notifications via various channels."""
    
    @staticmethod
    async def send_email_otp(email: str, otp_code: str) -> bool:
        """Send OTP via email."""
        # TODO: Integrate with SendGrid, AWS SES, or SMTP
        logger.info(f"📧 [EMAIL] Sending OTP to {email}: {otp_code}")
        
        # For development, just log it
        print(f"\n{'='*50}")
        print(f"📧 EMAIL OTP")
        print(f"To: {email}")
        print(f"OTP Code: {otp_code}")
        print(f"{'='*50}\n")
        
        return True
    
    @staticmethod
    async def send_sms_otp(phone: str, otp_code: str) -> bool:
        """Send OTP via SMS."""
        # TODO: Integrate with Twilio, MSG91, or AWS SNS
        logger.info(f"📱 [SMS] Sending OTP to {phone}: {otp_code}")
        
        print(f"\n{'='*50}")
        print(f"📱 SMS OTP")
        print(f"To: {phone}")
        print(f"OTP Code: {otp_code}")
        print(f"{'='*50}\n")
        
        return True
    
    @staticmethod
    async def send_whatsapp_otp(phone: str, otp_code: str) -> bool:
        """Send OTP via WhatsApp."""
        # TODO: Integrate with WhatsApp Business API
        logger.info(f"💬 [WhatsApp] Sending OTP to {phone}: {otp_code}")
        
        print(f"\n{'='*50}")
        print(f"💬 WhatsApp OTP")
        print(f"To: {phone}")
        print(f"OTP Code: {otp_code}")
        print(f"{'='*50}\n")
        
        return True