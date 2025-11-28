# utils/notification_service.py
import os
import logging
import asyncio
from typing import Optional
from functools import partial
from time import sleep

from dotenv import load_dotenv
import resend  # blocking SDK

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")
# Basic sanity check:
if not RESEND_API_KEY:
    logger.warning("RESEND_API_KEY not set — emails will fail until you set it.")

# configure SDK (global)
resend.api_key = RESEND_API_KEY


class NotificationService:
    """Send notifications via various channels (Resend for email)."""

    @staticmethod
    async def _run_blocking(fn, *args, **kwargs):
        """Run a blocking function in the default threadpool so async event loop isn't blocked."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

    @staticmethod
    def _send_email_blocking(to: str, subject: str, html: str) -> dict:
        """Blocking call to the resend SDK. Returns SDK response dict or raises."""
        return resend.Emails.send({
            "from": FROM_EMAIL,
            "to": to,
            "subject": subject,
            "html": html,
        })

    @classmethod
    async def send_email_otp(cls, email: str, otp_code: str, max_retries: int = 3) -> bool:
        """Send OTP via Resend (async wrapper). Returns True on success, False on permanent failure."""
        logger.info(f"📧 [EMAIL] Sending OTP to {email}")
        subject = "Your verification code"
        html = f"<p>Your OTP code is <strong>{otp_code}</strong>. It expires in 10 minutes.</p>"

        attempt = 0
        backoff_seconds = 1
        while attempt < max_retries:
            attempt += 1
            try:
                # call blocking SDK in threadpool
                resp = await cls._run_blocking(cls._send_email_blocking, email, subject, html)
                # SDK returns an object/dict with an id — treat non-exception as success
                logger.info(f"📧 [EMAIL] Sent OTP to {email} (attempt {attempt}) resp: {resp}")
                return True
            except Exception as exc:
                # log detailed error — the SDK raises exceptions for network/auth issues
                logger.exception(f"📧 [EMAIL] Error sending OTP to {email} (attempt {attempt}): {exc}")
                # simple exponential backoff
                if attempt < max_retries:
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds *= 2
                else:
                    logger.error(f"📧 [EMAIL] Failed to send OTP to {email} after {attempt} attempts.")
                    return False

    @staticmethod
    async def send_sms_otp(phone: str, otp_code: str) -> bool:
        """Stub for SMS — implement with Twilio/Msg91/Fast2SMS. Currently logs for dev."""
        logger.info(f"📱 [SMS] Sending OTP to {phone}: {otp_code}")
        print(f"\n{'='*50}")
        print(f"📱 SMS OTP\nTo: {phone}\nOTP Code: {otp_code}\n{'='*50}\n")
        return True

    @staticmethod
    async def send_whatsapp_otp(phone: str, otp_code: str) -> bool:
        """Stub for WhatsApp — implement later with appropriate provider."""
        logger.info(f"💬 [WhatsApp] Sending OTP to {phone}: {otp_code}")
        print(f"\n{'='*50}")
        print(f"💬 WhatsApp OTP\nTo: {phone}\nOTP Code: {otp_code}\n{'='*50}\n")
        return True
