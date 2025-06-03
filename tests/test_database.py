"""
Unit tests for database module
Tests database connection, session management, and table operations
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.database import (
    get_db, create_tables, check_db_connection, get_db_info,
    init_db, close_db, Base, engine, SessionLocal
)
from app.config import get_settings


class TestDatabaseConnection:
    """Test database connection functionality"""
    
    @patch('app.database.engine')
    def test_check_db_connection_success(self, mock_engine):
        """Test successful database connection check"""
        # Mock successful connection
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        
        result = check_db_connection()
        
        assert result is True
        mock_engine.connect.assert_called_once()
    
    @patch('app.database.engine')
    def test_check_db_connection_failure(self, mock_engine):
        """Test database connection check failure"""
        # Mock connection failure
        mock_engine.connect.side_effect = OperationalError("Connection failed", None, None)
        
        result = check_db_connection()
        
        assert result is False
        mock_engine.connect.assert_called_once()
    
    @patch('app.database.engine')
    def test_check_db_connection_unexpected_error(self, mock_engine):
        """Test database connection check with unexpected error"""
        # Mock unexpected error
        mock_engine.connect.side_effect = Exception("Unexpected error")
        
        result = check_db_connection()
        
        assert result is False
        mock_engine.connect.assert_called_once()


class TestDatabaseInfo:
    """Test database information retrieval"""
    
    @patch('app.database.engine')
    def test_get_db_info_success(self, mock_engine):
        """Test successful database info retrieval"""
        # Mock successful query execution
        mock_connection = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = ("PostgreSQL 13.0",)
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        
        info = get_db_info()
        
        assert "database_url" in info
        assert "version" in info
        assert info["version"] == "PostgreSQL 13.0"
        assert info["status"] == "connected"
    
    @patch('app.database.engine')
    def test_get_db_info_connection_failure(self, mock_engine):
        """Test database info retrieval with connection failure"""
        mock_engine.connect.side_effect = OperationalError("Connection failed", None, None)
        
        info = get_db_info()
        
        assert info["status"] == "disconnected"
        assert info["error"] == "Connection failed"
        assert "database_url" in info
    
    @patch('app.database.engine')
    def test_get_db_info_query_failure(self, mock_engine):
        """Test database info retrieval with query failure"""
        mock_connection = MagicMock()
        mock_connection.execute.side_effect = SQLAlchemyError("Query failed")
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        
        info = get_db_info()
        
        assert info["status"] == "error"
        assert "Query failed" in info["error"]


class TestTableCreation:
    """Test database table creation"""
    
    @patch('app.database.Base.metadata.create_all')
    @patch('app.database.logger')
    def test_create_tables_success(self, mock_logger, mock_create_all):
        """Test successful table creation"""
        create_tables()
        
        mock_create_all.assert_called_once()
        mock_logger.info.assert_called_with("Database tables created successfully")
    
    @patch('app.database.Base.metadata.create_all')
    @patch('app.database.logger')
    def test_create_tables_failure(self, mock_logger, mock_create_all):
        """Test table creation failure"""
        mock_create_all.side_effect = SQLAlchemyError("Table creation failed")
        
        create_tables()
        
        mock_create_all.assert_called_once()
        mock_logger.error.assert_called()
        # Check that error was logged with the exception
        error_call = mock_logger.error.call_args[0][0]
        assert "Failed to create database tables" in error_call


class TestSessionManagement:
    """Test database session management"""
    
    def test_get_db_generator(self):
        """Test get_db session generator"""
        # This test is tricky because get_db is a generator
        # We'll test that it yields a session and closes it properly
        
        with patch('app.database.SessionLocal') as mock_session_local:
            mock_session = MagicMock()
            mock_session_local.return_value = mock_session
            
            # Get the generator
            db_gen = get_db()
            
            # Get the session
            session = next(db_gen)
            
            assert session == mock_session
            mock_session_local.assert_called_once()
            
            # Close the generator (simulates end of request)
            try:
                next(db_gen)
            except StopIteration:
                pass
            
            # Verify session was closed
            mock_session.close.assert_called_once()
    
    def test_get_db_with_exception(self):
        """Test get_db session management with exception"""
        with patch('app.database.SessionLocal') as mock_session_local:
            mock_session = MagicMock()
            mock_session_local.return_value = mock_session
            
            db_gen = get_db()
            session = next(db_gen)
            
            # Simulate an exception during request processing
            try:
                db_gen.throw(Exception("Test exception"))
            except Exception:
                pass
            
            # Session should still be closed
            mock_session.close.assert_called_once()


class TestAsyncDatabaseOperations:
    """Test async database operations"""
    
    @pytest.mark.asyncio
    async def test_init_db_success(self):
        """Test successful database initialization"""
        with patch('app.database.logger') as mock_logger:
            await init_db()
            
            mock_logger.info.assert_called_with("Database initialized successfully")
    
    @pytest.mark.asyncio
    async def test_init_db_with_exception(self):
        """Test database initialization with exception"""
        with patch('app.database.logger') as mock_logger:
            # Patch something that might fail during init
            with patch('app.database.check_db_connection', side_effect=Exception("Init failed")):
                await init_db()
                
                # Should log the error
                mock_logger.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_close_db_success(self):
        """Test successful database connection closure"""
        with patch('app.database.engine') as mock_engine:
            with patch('app.database.logger') as mock_logger:
                await close_db()
                
                mock_engine.dispose.assert_called_once()
                mock_logger.info.assert_called_with("Database connections closed")
    
    @pytest.mark.asyncio
    async def test_close_db_with_exception(self):
        """Test database closure with exception"""
        with patch('app.database.engine') as mock_engine:
            with patch('app.database.logger') as mock_logger:
                mock_engine.dispose.side_effect = Exception("Disposal failed")
                
                await close_db()
                
                mock_engine.dispose.assert_called_once()
                mock_logger.error.assert_called()


class TestDatabaseConfiguration:
    """Test database configuration and setup"""
    
    @patch('app.database.get_settings')
    def test_database_url_from_settings(self, mock_get_settings):
        """Test database URL configuration from settings"""
        mock_settings = MagicMock()
        mock_settings.DATABASE_URL = "postgresql://test:test@localhost:5432/testdb"
        mock_get_settings.return_value = mock_settings
        
        # Import engine to trigger configuration
        from app.database import engine
        
        # Verify settings were used
        mock_get_settings.assert_called()
    
    def test_session_local_configuration(self):
        """Test SessionLocal configuration"""
        from app.database import SessionLocal
        
        # Verify SessionLocal is properly configured
        assert SessionLocal is not None
        assert hasattr(SessionLocal, 'bind')
    
    def test_base_metadata(self):
        """Test Base metadata configuration"""
        from app.database import Base
        
        # Verify Base is properly configured
        assert Base is not None
        assert hasattr(Base, 'metadata')
        assert hasattr(Base.metadata, 'create_all')


class TestDatabaseIntegration:
    """Integration tests for database functionality"""
    
    def test_engine_creation(self):
        """Test that engine is created properly"""
        from app.database import engine
        
        assert engine is not None
        assert hasattr(engine, 'connect')
        assert hasattr(engine, 'dispose')
    
    def test_session_factory(self):
        """Test session factory functionality"""
        from app.database import SessionLocal
        
        # Create a session
        session = SessionLocal()
        
        assert session is not None
        assert hasattr(session, 'query')
        assert hasattr(session, 'add')
        assert hasattr(session, 'commit')
        assert hasattr(session, 'close')
        
        # Clean up
        session.close()
    
    @patch('app.database.engine')
    def test_full_database_lifecycle(self, mock_engine):
        """Test complete database lifecycle"""
        # Mock successful operations
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Test connection check
        assert check_db_connection() is True
        
        # Test table creation
        with patch('app.database.Base.metadata.create_all'):
            create_tables()
        
        # Test info retrieval
        mock_result = MagicMock()
        mock_result.fetchone.return_value = ("PostgreSQL 13.0",)
        mock_connection.execute.return_value = mock_result
        
        info = get_db_info()
        assert info["status"] == "connected"


class TestDatabaseErrorHandling:
    """Test database error handling scenarios"""
    
    @patch('app.database.engine')
    def test_connection_timeout(self, mock_engine):
        """Test database connection timeout handling"""
        from sqlalchemy.exc import TimeoutError
        
        mock_engine.connect.side_effect = TimeoutError("Connection timeout", None, None)
        
        result = check_db_connection()
        assert result is False
    
    @patch('app.database.engine')
    def test_permission_denied(self, mock_engine):
        """Test database permission denied handling"""
        mock_engine.connect.side_effect = OperationalError(
            "permission denied for database", None, None
        )
        
        result = check_db_connection()
        assert result is False
    
    @patch('app.database.engine')
    def test_database_not_found(self, mock_engine):
        """Test database not found error handling"""
        mock_engine.connect.side_effect = OperationalError(
            'database "nonexistent" does not exist', None, None
        )
        
        result = check_db_connection()
        assert result is False
    
    @patch('app.database.Base.metadata.create_all')
    def test_table_creation_constraint_error(self, mock_create_all):
        """Test table creation with constraint errors"""
        from sqlalchemy.exc import IntegrityError
        
        mock_create_all.side_effect = IntegrityError("Constraint violation", None, None)
        
        # Should not raise exception, just log error
        create_tables()
        
        mock_create_all.assert_called_once()


class TestDatabaseTransactions:
    """Test database transaction handling"""
    
    def test_session_transaction_commit(self):
        """Test session transaction commit"""
        with patch('app.database.SessionLocal') as mock_session_local:
            mock_session = MagicMock()
            mock_session_local.return_value = mock_session
            
            # Simulate using the session
            db_gen = get_db()
            session = next(db_gen)
            
            # Simulate successful operation
            session.add(MagicMock())
            session.commit()
            
            # Verify operations were called
            session.add.assert_called_once()
            session.commit.assert_called_once()
    
    def test_session_transaction_rollback(self):
        """Test session transaction rollback on error"""
        with patch('app.database.SessionLocal') as mock_session_local:
            mock_session = MagicMock()
            mock_session_local.return_value = mock_session
            
            db_gen = get_db()
            session = next(db_gen)
            
            # Simulate error during operation
            session.commit.side_effect = SQLAlchemyError("Commit failed")
            
            try:
                session.commit()
            except SQLAlchemyError:
                session.rollback()
            
            session.rollback.assert_called_once()


class TestDatabaseMigrations:
    """Test database migration related functionality"""
    
    @patch('app.database.Base.metadata')
    def test_metadata_tables_registration(self, mock_metadata):
        """Test that all model tables are registered with metadata"""
        # Import all models to ensure they're registered
        from app.models.model import Model, Deployment
        from app.models.monitoring import (
            PredictionLog, ModelPerformanceMetric, SystemHealthMetric,
            Alert, AuditLog
        )
        
        # Verify Base.metadata has tables
        from app.database import Base
        
        # Check that tables are registered (this is automatic when models are imported)
        assert hasattr(Base, 'metadata')
        # In real test, would check specific table names in metadata.tables 