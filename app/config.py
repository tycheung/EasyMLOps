"""
Configuration management for EasyMLOps platform
Handles environment variables, database settings, and application configuration
"""

import os
from typing import Optional
from pydantic import PostgresDsn, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Settings
    APP_NAME: str = "EasyMLOps"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    
    # Database Settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "easymlops"
    DATABASE_URL: Optional[PostgresDsn] = None
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict[str, any]) -> any:
        if isinstance(v, str):
            return v
        
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        host = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT")
        db = values.get("POSTGRES_DB")
        
        # Build PostgreSQL URL manually for Pydantic v2 compatibility
        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{db}"
        else:
            return f"postgresql://{user}@{host}:{port}/{db}"
    
    # File Storage Settings
    MODELS_DIR: str = "models"
    BENTOS_DIR: str = "bentos" 
    STATIC_DIR: str = "static"
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_MODEL_EXTENSIONS: list[str] = [".pkl", ".joblib", ".h5", ".pb", ".onnx", ".json"]
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # BentoML Settings
    BENTOML_HOME: str = "bentos"
    
    # Monitoring Settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


# Create necessary directories
def create_directories():
    """Create required directories if they don't exist"""
    directories = [
        settings.MODELS_DIR,
        settings.BENTOS_DIR,
        settings.STATIC_DIR,
        "logs",
        "alembic/versions"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True) 