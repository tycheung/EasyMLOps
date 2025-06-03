"""
Database connection and session management for EasyMLOps
Handles PostgreSQL and SQLite connections with SQLModel, session lifecycle, and database utilities
"""

from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
from typing import Generator, AsyncGenerator
from contextlib import asynccontextmanager
import os

from app.config import get_settings

# Get settings
settings = get_settings()

# Configure logging
logger = logging.getLogger(__name__)

def create_database_engine():
    """Create database engine based on configuration"""
    database_url = str(settings.DATABASE_URL)
    
    if settings.is_sqlite():
        # SQLite configuration
        # Ensure the directory exists for the SQLite file
        sqlite_path = settings.SQLITE_PATH
        sqlite_dir = os.path.dirname(sqlite_path)
        if sqlite_dir and not os.path.exists(sqlite_dir):
            os.makedirs(sqlite_dir, exist_ok=True)
        
        return create_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,  # Allow SQLite to be used in multiple threads
                "timeout": 20  # Timeout for database operations
            },
            echo=settings.DEBUG,  # Log SQL queries in debug mode
        )
    else:
        # PostgreSQL configuration
        return create_engine(
            database_url,
            poolclass=QueuePool,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
            echo=settings.DEBUG,  # Log SQL queries in debug mode
        )

def create_async_database_engine():
    """Create async database engine based on configuration"""
    if settings.is_sqlite():
        # SQLite async using aiosqlite
        database_url = str(settings.DATABASE_URL).replace('sqlite:///', 'sqlite+aiosqlite:///')
        return create_async_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=settings.DEBUG,
        )
    else:
        # PostgreSQL async using asyncpg
        async_database_url = str(settings.DATABASE_URL).replace('postgresql://', 'postgresql+asyncpg://')
        return create_async_engine(
            async_database_url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=settings.DEBUG,
        )

# Create database engines
engine = create_database_engine()
async_engine = create_async_database_engine()

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Compatibility exports for existing models and tests
# For SQLAlchemy-style models
Base = declarative_base()

# For traditional SQLAlchemy sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# For SQLModel (recommended for new models)
SQLModelBase = SQLModel

def get_db() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI
    Creates a new database session for each request
    """
    with Session(engine) as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            session.rollback()
            raise


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async database session context manager
    Creates a new async database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Async database session error: {e}")
            await session.rollback()
            raise


async def init_db():
    """Initialize database tables"""
    try:
        # Initialize SQLite database file if needed
        if settings.is_sqlite():
            from app.config import init_sqlite_database
            init_sqlite_database()
        
        # Import all models to ensure they're registered with metadata
        from app.models.model import Model, ModelDeployment
        from app.models.monitoring import (
            PredictionLogDB, ModelPerformanceMetricsDB, SystemHealthMetricDB,
            AlertDB, AuditLogDB
        )
        
        # Create tables synchronously (SQLModel doesn't support async table creation yet)
        # Create SQLModel tables
        SQLModel.metadata.create_all(engine)
        # Create SQLAlchemy Base tables
        Base.metadata.create_all(engine)
        logger.info(f"Database initialized successfully using {settings.get_db_type()}")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def close_db():
    """Close database connections"""
    try:
        if not settings.is_sqlite():
            # Only dispose async engine for PostgreSQL
            await async_engine.dispose()
        engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


def create_tables():
    """Create all database tables"""
    try:
        # Initialize SQLite database file if needed
        if settings.is_sqlite():
            from app.config import init_sqlite_database
            init_sqlite_database()
        
        # Import all models to ensure they're registered with metadata
        from app.models.model import Model, ModelDeployment
        from app.models.monitoring import (
            PredictionLogDB, ModelPerformanceMetricsDB, SystemHealthMetricDB,
            AlertDB, AuditLogDB
        )
        
        # Create SQLModel tables
        SQLModel.metadata.create_all(engine)
        # Create SQLAlchemy Base tables  
        Base.metadata.create_all(engine)
        logger.info(f"Database tables created successfully using {settings.get_db_type()}")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def drop_tables():
    """Drop all database tables (use with caution!)"""
    try:
        # Drop SQLModel tables
        SQLModel.metadata.drop_all(engine)
        # Drop SQLAlchemy Base tables
        Base.metadata.drop_all(engine)
        logger.warning("All database tables dropped")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise


def check_db_connection() -> bool:
    """Check if database connection is working"""
    try:
        with engine.connect() as connection:
            if settings.is_sqlite():
                connection.exec_driver_sql("SELECT 1")
            else:
                connection.exec_driver_sql("SELECT 1")
        logger.info(f"Database connection successful ({settings.get_db_type()})")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def check_async_db_connection() -> bool:
    """Check if async database connection is working"""
    try:
        async with async_engine.connect() as connection:
            await connection.exec_driver_sql("SELECT 1")
        logger.info(f"Async database connection successful ({settings.get_db_type()})")
        return True
    except Exception as e:
        logger.error(f"Async database connection failed: {e}")
        return False


class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    def get_session() -> Session:
        """Get a new database session"""
        return Session(engine)
    
    @staticmethod
    def close_session(db: Session):
        """Close database session"""
        try:
            db.close()
        except Exception as e:
            logger.error(f"Error closing database session: {e}")
    
    @staticmethod
    def commit_session(db: Session):
        """Commit database session"""
        try:
            db.commit()
        except Exception as e:
            logger.error(f"Error committing database session: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def rollback_session(db: Session):
        """Rollback database session"""
        try:
            db.rollback()
        except Exception as e:
            logger.error(f"Error rolling back database session: {e}")


# Database health check
def get_db_info() -> dict:
    """Get database connection information"""
    try:
        with engine.connect() as connection:
            if settings.is_sqlite():
                # SQLite version query
                result = connection.exec_driver_sql("SELECT sqlite_version()")
                version = f"SQLite {result.fetchone()[0]}"
                
                return {
                    "status": "connected",
                    "database_type": "SQLite",
                    "database_path": settings.SQLITE_PATH,
                    "version": version,
                    "file_exists": os.path.exists(settings.SQLITE_PATH),
                    "file_size": os.path.getsize(settings.SQLITE_PATH) if os.path.exists(settings.SQLITE_PATH) else 0
                }
            else:
                # PostgreSQL version query
                result = connection.exec_driver_sql("SELECT version()")
                version = result.fetchone()[0]
                
                # Hide password in URL
                safe_url = str(settings.DATABASE_URL)
                if hasattr(settings, 'POSTGRES_PASSWORD') and settings.POSTGRES_PASSWORD:
                    safe_url = safe_url.replace(settings.POSTGRES_PASSWORD, "***")
                
                return {
                    "status": "connected",
                    "database_type": "PostgreSQL",
                    "database_url": safe_url,
                    "version": version,
                    "pool_size": engine.pool.size(),
                    "checked_out_connections": engine.pool.checkedout()
                }
    except Exception as e:
        # Hide password in URL for error reporting
        safe_url = str(settings.DATABASE_URL)
        if hasattr(settings, 'POSTGRES_PASSWORD') and settings.POSTGRES_PASSWORD:
            safe_url = safe_url.replace(settings.POSTGRES_PASSWORD, "***")
        
        return {
            "status": "disconnected",
            "database_type": settings.get_db_type(),
            "error": str(e),
            "database_url": safe_url if not settings.is_sqlite() else settings.SQLITE_PATH
        } 