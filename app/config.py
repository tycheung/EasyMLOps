"""
Configuration management for EasyMLOps platform
Handles environment variables, database settings, and application configuration
"""

import os
from typing import Optional, Union
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
    USE_SQLITE: bool = False  # New flag to choose database type
    SQLITE_PATH: str = "easymlops.db"  # SQLite database file path
    
    # PostgreSQL Settings (used when USE_SQLITE=False)
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "easymlops"
    DATABASE_URL: Optional[Union[PostgresDsn, str]] = None
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict[str, any]) -> any:
        if isinstance(v, str):
            return v
        
        # Check if we should use SQLite
        use_sqlite = values.get("USE_SQLITE", False)
        
        if use_sqlite:
            sqlite_path = values.get("SQLITE_PATH", "easymlops.db")
            return f"sqlite:///{sqlite_path}"
        
        # Build PostgreSQL URL
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
    
    def is_sqlite(self) -> bool:
        """Check if using SQLite database"""
        return self.USE_SQLITE
    
    def get_db_type(self) -> str:
        """Get database type string"""
        return "SQLite" if self.USE_SQLITE else "PostgreSQL"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


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


def init_sqlite_database():
    """Initialize SQLite database if it doesn't exist"""
    if settings.is_sqlite():
        sqlite_path = settings.SQLITE_PATH
        if not os.path.exists(sqlite_path):
            # Create empty SQLite database file
            import sqlite3
            conn = sqlite3.connect(sqlite_path)
            conn.close()
            print(f"Created SQLite database: {sqlite_path}")
        else:
            print(f"Using existing SQLite database: {sqlite_path}") 