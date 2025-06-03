"""
Database connection and session management for EasyMLOps
Handles PostgreSQL connection with SQLModel, session lifecycle, and database utilities
"""

from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import logging
from typing import Generator, AsyncGenerator
from contextlib import asynccontextmanager

from app.config import get_settings

# Get settings
settings = get_settings()

# Configure logging
logger = logging.getLogger(__name__)

# Create SQLModel engine
engine = create_engine(
    str(settings.DATABASE_URL),
    poolclass=StaticPool,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.DEBUG,  # Log SQL queries in debug mode
)

# Create async engine for async operations
async_database_url = str(settings.DATABASE_URL).replace('postgresql://', 'postgresql+asyncpg://')
async_engine = create_async_engine(
    async_database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.DEBUG,
)

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


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
        # Create tables synchronously (SQLModel doesn't support async table creation yet)
        SQLModel.metadata.create_all(engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def close_db():
    """Close database connections"""
    try:
        await async_engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


def create_tables():
    """Create all database tables"""
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def drop_tables():
    """Drop all database tables (use with caution!)"""
    try:
        SQLModel.metadata.drop_all(engine)
        logger.warning("All database tables dropped")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise


def check_db_connection() -> bool:
    """Check if database connection is working"""
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def check_async_db_connection() -> bool:
    """Check if async database connection is working"""
    try:
        async with async_engine.connect() as connection:
            await connection.exec_driver_sql("SELECT 1")
        logger.info("Async database connection successful")
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
            result = connection.exec_driver_sql("SELECT version()")
            version = result.fetchone()[0]
            return {
                "status": "connected",
                "database_url": str(settings.DATABASE_URL).replace(settings.POSTGRES_PASSWORD, "***"),
                "version": version,
                "pool_size": engine.pool.size(),
                "checked_out_connections": engine.pool.checkedout()
            }
    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e),
            "database_url": str(settings.DATABASE_URL).replace(settings.POSTGRES_PASSWORD, "***")
        } 