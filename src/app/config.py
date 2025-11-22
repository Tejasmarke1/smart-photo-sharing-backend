from pydantic import BaseSettings,Field,PostgresDsn,EmailStr
from typing import List
from functools import lru_cache
import os

class Settings(BaseSettings):
    APP_Name: str = Field(..., env="APP_NAME")
    ENVIRONMENT : str = Field(..., env="ENVIRONMENT")
    DEBUG: bool = Field(True, env="DEBUG")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field([], env="BACKEND_CORS_ORIGINS")
    # API prefix
    
    API_V1_PREFIX: str = "/api/v1"
    
    # Database
    DATABASE_URL: PostgresDsn = Field(..., env="DATABASE_URL")
    
    
    # Redis
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    
    # JWT
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    
    # Email
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[EmailStr] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM: Optional[EmailStr] = None
    EMAIL_TEMPLATES_DIR: str = "./app/email_templates"
    
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        

@lru_cache()
def get_settings():
    return Settings()


Settings = get_settings()

    
    