"""Application configuration using Pydantic Settings."""
from typing import Optional, List
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

import warnings
warnings.filterwarnings('ignore', message='.*MINGW.*')


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    APP_NAME: str = "backend"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me-in-production"
    API_V1_PREFIX: str = "/api/v1"
    
    # Database
    DB_USER: str = "backend"
    DB_PASSWORD: str = "backend"
    DB_HOST: str = "127.0.0.1"
    DB_PORT: str = "5433"
    DB_NAME: str = "backend"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    
    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL from components."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    
    #REsend Email Service
    RESEND_API_KEY: str = "re_Z85ezC4T_EHzN9W7Asr7XLxeuBqChH4J3"
    RESEND_EMAIL_FROM: str = "onboarding@resend.dev"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    REDIS_DB: str = "0"
    CELERY_BROKER_DB: str = "1"
    CELERY_RESULT_DB: str = "2"
    
    @computed_field
    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL."""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @computed_field
    @property
    def CELERY_BROKER_URL(self) -> str:
        """Construct Celery broker URL."""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.CELERY_BROKER_DB}"
    
    @computed_field
    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        """Construct Celery result backend URL."""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.CELERY_RESULT_DB}"
    
    # AWS/S3
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET: str = "backend-photos"
    S3_ENDPOINT_URL: Optional[str] = None
    CDN_BASE_URL: Optional[str] = None
    
    # Authentication
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Payments (Razorpay)
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""
    RAZORPAY_WEBHOOK_SECRET: str = ""
    
    # WhatsApp
    WHATSAPP_API_KEY: str = ""
    WHATSAPP_PHONE_NUMBER_ID: str = ""
    
    # Face Detection/Recognition
    FACE_DETECTION_MODEL: str = "retinaface"
    FACE_EMBEDDING_MODEL: str = "arcface"
    EMBEDDING_DIMENSION: int = 512
    FACE_CONFIDENCE_THRESHOLD: float = 0.9
    FACE_MATCH_THRESHOLD: float = 0.6
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    ENABLE_PROMETHEUS: bool = True
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    UPLOAD_RATE_LIMIT: int = 10
    SEARCH_RATE_LIMIT: int = 20


settings = Settings()
