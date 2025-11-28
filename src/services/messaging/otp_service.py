"""OTP generation and verification service."""
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session
import logging

from src.models.otp import OTP
from src.core.security import generate_otp

logger = logging.getLogger(__name__)


class OTPService:
    """Handle OTP operations."""
    
    @staticmethod
    def create_otp(
        db: Session,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        otp_type: str = "login",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[OTP, str]:
        """Create and return OTP."""
        # Invalidate previous OTPs
        query = db.query(OTP)
        if email:
            query = query.filter(OTP.email == email)
        if phone:
            query = query.filter(OTP.phone == phone)
        
        query.filter(OTP.is_verified == False).update({"is_verified": True})
        db.commit()
        
        # Generate new OTP
        otp_code = generate_otp()
        expires_at = datetime.utcnow() + timedelta(minutes=5)
        
        otp = OTP(
            email=email,
            phone=phone,
            otp_code=otp_code,
            otp_type=otp_type,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        db.add(otp)
        db.commit()
        db.refresh(otp)
        
        logger.info(f"OTP created for {email or phone}: {otp_code}")
        return otp, otp_code
    
    @staticmethod
    def verify_otp(
        db: Session,
        email: Optional[str],
        phone: Optional[str],
        otp_code: str
    ) -> Tuple[bool, Optional[str]]:
        """Verify OTP and return success status and message."""
        query = db.query(OTP)
        if email:
            query = query.filter(OTP.email == email)
        elif phone:
            query = query.filter(OTP.phone == phone)
        else:
            return False, "Email or phone required"
        
        otp = query.filter(
            OTP.otp_code == otp_code,
            OTP.is_verified == False
        ).order_by(OTP.created_at.desc()).first()
        
        if not otp:
            return False, "Invalid OTP"
        
        if otp.is_expired():
            return False, "OTP expired"
        
        if otp.attempts >= otp.max_attempts:
            return False, "Maximum attempts exceeded"
        
        if otp.otp_code != otp_code:
            otp.increment_attempts()
            db.commit()
            return False, f"Invalid OTP. {otp.max_attempts - otp.attempts} attempts remaining"
        
        # Mark as verified
        otp.is_verified = True
        otp.verified_at = datetime.utcnow()
        db.commit()
        
        return True, "OTP verified successfully"