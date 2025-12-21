"""Application configuration using Pydantic Settings (ENV ONLY)."""
from typing import Optional
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings (MUST come from environment variables)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = Field(..., env="APP_NAME")
    ENVIRONMENT: str = Field(..., env="ENVIRONMENT")
    DEBUG: bool = Field(..., env="DEBUG")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    API_V1_PREFIX: str = Field(..., env="API_V1_PREFIX")

    # Database
    DB_USER: str = Field(..., env="DB_USER")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")
    DB_HOST: str = Field(..., env="DB_HOST")
    DB_PORT: int = Field(..., env="DB_PORT")
    DB_NAME: str = Field(..., env="DB_NAME")
    DB_POOL_SIZE: int = Field(..., env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(..., env="DB_MAX_OVERFLOW")

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # Redis / Celery
    REDIS_HOST: str = Field(..., env="REDIS_HOST")
    REDIS_PORT: int = Field(..., env="REDIS_PORT")
    REDIS_URL: str = Field(..., env="REDIS_URL")

    CELERY_BROKER_DB: int = Field(..., env="CELERY_BROKER_DB")
    CELERY_RESULT_DB: int = Field(..., env="CELERY_RESULT_DB")

    # @computed_field
    # @property
    # def REDIS_URL(self) -> str:
    #     return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @computed_field
    @property
    def CELERY_BROKER_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.CELERY_BROKER_DB}"

    @computed_field
    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.CELERY_RESULT_DB}"
    
    
    
    
    # AWS / S3
    AWS_ACCESS_KEY_ID: str = Field(..., env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    S3_BUCKET_NAME: str = Field(..., env="S3_BUCKET_NAME")  
    S3_REGION: str = Field(..., env="S3_REGION")    
    # S3_ENDPOINT_URL: Optional[str] = Field(None, env="S3_ENDPOINT_URL")
    
    
    # CDN_URL: Optional[str] = Field(None, env="CDN_URL")
    
    
    JWT_ALGORITHM: str = Field(..., env="JWT_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(..., env="ACCESS_TOKEN_EXPIRE_MINUTES")  
    REFRESH_TOKEN_EXPIRE_MINUTES: int = Field(..., env="REFRESH_TOKEN_EXPIRE_MINUTES")
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")
    
    
    
    #RESEND 
    RESEND_API_KEY: str = Field(..., env="RESEND_API_KEY")
    RESEND_FROM_EMAIL: str = Field(..., env="RESEND_FROM_EMAIL")
    
    
    #razorpay
    RAZORPAY_KEY_ID: str = Field(..., env="RAZORPAY_KEY_ID")
    RAZORPAY_KEY_SECRET: str = Field(..., env="RAZORPAY_KEY_SECRET")
    RAZORPAY_WEBHOOK_SECRET: str = Field(..., env="RAZORPAY_WEBHOOK_SECRET")
    
    # WhatsApp
    WHATSAPP_API_KEY: str = Field(..., env="WHATSAPP_API_KEY")
    WHATSAPP_PHONE_NUMBER_ID: str = Field(..., env="WHATSAPP_PHONE_NUMBER_ID")

    # Face recognition
    FACE_DETECTION_MODEL: str = Field(..., env="FACE_DETECTION_MODEL")
    FACE_EMBEDDING_MODEL: str = Field(..., env="FACE_EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(..., env="EMBEDDING_DIMENSION")
    FACE_CONFIDENCE_THRESHOLD: float = Field(..., env="FACE_CONFIDENCE_THRESHOLD")
    FACE_MATCH_THRESHOLD: float = Field(..., env="FACE_MATCH_THRESHOLD")

    # Monitoring
    SENTRY_DSN: Optional[str] = Field(None, env="SENTRY_DSN")
    ENABLE_PROMETHEUS: bool = Field(..., env="ENABLE_PROMETHEUS")

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = Field(..., env="RATE_LIMIT_PER_MINUTE")
    UPLOAD_RATE_LIMIT: int = Field(..., env="UPLOAD_RATE_LIMIT")
    SEARCH_RATE_LIMIT: int = Field(..., env="SEARCH_RATE_LIMIT")
    
    OTP_SEND_LIMIT: int = Field(..., env="OTP_SEND_LIMIT")
    OTP_VERIFY_ATTEMPTS: int = Field(..., env="OTP_VERIFY_ATTEMPTS")
    OTP_SEND_WINDOW_SECS: int = Field(..., env="OTP_SEND_WINDOW_SECS")
    OTP_VERIFY_LOCK_SECS: int = Field(..., env="OTP_VERIFY_LOCK_SECS")
    
    FACE_DEVICE: str = Field(..., env="FACE_DEVICE")  


settings = Settings()
    