"""
Application startup tests
Tests application initialization, configuration, database setup, and lifespan events
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from fastapi.exceptions import RequestValidationError

from app.core.app_factory import create_app, create_lifespan, RequestLoggingMiddleware
from app.main import parse_arguments, setup_application_config, configure_database_mode


class TestApplicationSetup:
    """Test application creation and configuration"""
    
    def test_create_app_basic(self):
        """Test basic app creation"""
        with patch('app.core.app_factory.Settings') as mock_settings_class:
            mock_settings = MagicMock()
            mock_settings.APP_NAME = "EasyMLOps"
            mock_settings.APP_VERSION = "1.0.0"
            mock_settings.BACKEND_CORS_ORIGINS = ["http://localhost:3000"]
            mock_settings.STATIC_DIR = "static"
            mock_settings.API_V1_PREFIX = "/api/v1"
            mock_settings.DEBUG = False
            mock_settings.is_sqlite.return_value = True
            mock_settings.get_db_type.return_value = "SQLite"
            
            with patch('os.path.exists', return_value=False):
                test_app = create_app(mock_settings)
            
            assert test_app is not None
            assert test_app.title == "EasyMLOps"
            assert test_app.version == "1.0.0"
    
    def test_create_app_middleware_configuration(self):
        """Test middleware is properly configured"""
        mock_settings = MagicMock()
        mock_settings.APP_NAME = "EasyMLOps"
        mock_settings.APP_VERSION = "1.0.0"
        mock_settings.BACKEND_CORS_ORIGINS = ["http://localhost:3000"]
        mock_settings.STATIC_DIR = "static"
        mock_settings.API_V1_PREFIX = "/api/v1"
        mock_settings.DEBUG = False
        mock_settings.is_sqlite.return_value = True
        mock_settings.get_db_type.return_value = "SQLite"
        
        with patch('os.path.exists', return_value=False):
            test_app = create_app(mock_settings)
        
        middleware_details = []
        for middleware in test_app.user_middleware:
            if hasattr(middleware, 'cls'):
                middleware_details.append(str(middleware.cls))
            else:
                middleware_details.append(str(type(middleware)))
        
        assert any("RequestLoggingMiddleware" in detail for detail in middleware_details)
        assert any("GZip" in detail for detail in middleware_details)
        assert any("CORS" in detail for detail in middleware_details)
    
    def test_exception_handlers_registered(self):
        """Test exception handlers are properly registered"""
        mock_settings = MagicMock()
        mock_settings.APP_NAME = "EasyMLOps"
        mock_settings.APP_VERSION = "1.0.0"
        mock_settings.BACKEND_CORS_ORIGINS = ["http://localhost:3000"]
        mock_settings.STATIC_DIR = "static"
        mock_settings.API_V1_PREFIX = "/api/v1"
        mock_settings.DEBUG = False
        mock_settings.is_sqlite.return_value = True
        mock_settings.get_db_type.return_value = "SQLite"
        
        with patch('os.path.exists', return_value=False):
            test_app = create_app(mock_settings)
        
        from starlette.exceptions import HTTPException as StarletteHTTPException
        assert StarletteHTTPException in test_app.exception_handlers
        assert RequestValidationError in test_app.exception_handlers
        assert Exception in test_app.exception_handlers


class TestLifespanEvents:
    """Test application lifespan events"""
    
    @pytest.mark.asyncio
    @patch('app.database.check_async_db_connection')
    @patch('app.database.init_db')
    @patch('app.database.close_db')
    async def test_lifespan_startup_success(self, mock_close_db, mock_init_db, mock_check_db):
        """Test successful lifespan startup"""
        mock_check_db.return_value = True
        
        mock_settings = MagicMock()
        mock_settings.get_db_type.return_value = "SQLite"
        mock_settings.is_sqlite.return_value = True
        mock_logger = MagicMock()
        
        from fastapi import FastAPI
        test_app = FastAPI()
        
        async with create_lifespan(mock_settings, mock_logger)(test_app):
            pass
        
        mock_check_db.assert_called_once()
        mock_init_db.assert_called_once()
        mock_logger.info.assert_called()
        assert any("Starting EasyMLOps application" in str(call) for call in mock_logger.info.call_args_list)
        mock_close_db.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.database.check_async_db_connection')
    @patch('app.database.init_db')
    @patch('app.database.close_db')
    async def test_lifespan_database_connection_failure(self, mock_close_db, mock_init_db, mock_check_db):
        """Test lifespan with database connection failure"""
        mock_check_db.return_value = False
        
        mock_settings = MagicMock()
        mock_settings.get_db_type.return_value = "PostgreSQL"
        mock_settings.is_sqlite.return_value = False
        mock_logger = MagicMock()
        
        from fastapi import FastAPI
        test_app = FastAPI()
        
        async with create_lifespan(mock_settings, mock_logger)(test_app):
            pass
        
        mock_logger.error.assert_called()
        assert any("Database connection failed" in str(call) for call in mock_logger.error.call_args_list)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.start_monitoring_tasks')
    @patch('app.database.check_async_db_connection')
    @patch('app.database.init_db')
    @patch('app.database.close_db')
    async def test_lifespan_monitoring_service_startup(self, mock_close_db, mock_init_db,
                                                      mock_check_db, mock_start_monitoring):
        """Test lifespan with monitoring service startup"""
        mock_check_db.return_value = True
        mock_start_monitoring.return_value = None
        
        mock_settings = MagicMock()
        mock_settings.get_db_type.return_value = "SQLite"
        mock_settings.is_sqlite.return_value = True
        mock_logger = MagicMock()
        
        from fastapi import FastAPI
        test_app = FastAPI()
        
        with patch.dict(os.environ, {"DISABLE_MONITORING": "false"}, clear=False):
            async with create_lifespan(mock_settings, mock_logger)(test_app):
                pass
        
        mock_start_monitoring.assert_called_once()
        mock_logger.info.assert_called()
        assert any("Monitoring service started" in str(call) for call in mock_logger.info.call_args_list)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.start_monitoring_tasks')
    @patch('app.database.check_async_db_connection')
    @patch('app.database.init_db')
    @patch('app.database.close_db')
    async def test_lifespan_monitoring_disabled(self, mock_close_db, mock_init_db,
                                              mock_check_db, mock_start_monitoring):
        """Test lifespan with monitoring disabled"""
        mock_check_db.return_value = True
        
        mock_settings = MagicMock()
        mock_settings.get_db_type.return_value = "SQLite"
        mock_settings.is_sqlite.return_value = True
        mock_logger = MagicMock()
        
        from fastapi import FastAPI
        test_app = FastAPI()
        
        with patch.dict(os.environ, {"DISABLE_MONITORING": "true"}, clear=False):
            async with create_lifespan(mock_settings, mock_logger)(test_app):
                pass
        
        mock_start_monitoring.assert_not_called()
        mock_logger.info.assert_called()
        assert any("Monitoring service disabled" in str(call) for call in mock_logger.info.call_args_list)


class TestArgumentParsing:
    """Test command line argument parsing"""
    
    def test_parse_arguments_default(self):
        """Test parsing default arguments"""
        with patch('sys.argv', ['main.py']):
            args = parse_arguments()
            
            assert args.demo is False
            assert args.sqlite is False
            assert args.db_path == "easymlops.db"
            assert args.host is None
            assert args.port is None
            assert args.debug is False
            assert args.no_browser is False
    
    def test_parse_arguments_demo_mode(self):
        """Test parsing demo mode arguments"""
        with patch('sys.argv', ['main.py', '--demo']):
            args = parse_arguments()
            
            assert args.demo is True
            assert args.sqlite is False
    
    def test_parse_arguments_sqlite_mode(self):
        """Test parsing sqlite mode arguments"""
        with patch('sys.argv', ['main.py', '--sqlite', '--db-path', 'custom.db']):
            args = parse_arguments()
            
            assert args.demo is False
            assert args.sqlite is True
            assert args.db_path == "custom.db"
    
    def test_parse_arguments_server_options(self):
        """Test parsing server configuration arguments"""
        with patch('sys.argv', ['main.py', '--host', '127.0.0.1', '--port', '9000', '--debug']):
            args = parse_arguments()
            
            assert args.host == "127.0.0.1"
            assert args.port == 9000
            assert args.debug is True
    
    def test_parse_arguments_no_browser(self):
        """Test parsing no-browser argument"""
        with patch('sys.argv', ['main.py', '--no-browser']):
            args = parse_arguments()
            
            assert args.no_browser is True


class TestDatabaseConfiguration:
    """Test database configuration setup"""
    
    def test_configure_database_mode_demo(self):
        """Test configuring demo database mode"""
        mock_args = MagicMock()
        mock_args.demo = True
        mock_args.sqlite = False
        mock_args.db_path = "easymlops.db"
        
        with patch.dict(os.environ, {}, clear=True):
            is_demo = configure_database_mode(mock_args)
            
            assert is_demo is True
            assert os.environ.get("USE_SQLITE") == "true"
            assert os.environ.get("SQLITE_PATH") == "easymlops.db"
    
    def test_configure_database_mode_sqlite(self):
        """Test configuring sqlite database mode"""
        mock_args = MagicMock()
        mock_args.demo = False
        mock_args.sqlite = True
        mock_args.db_path = "custom.db"
        
        with patch.dict(os.environ, {}, clear=True):
            is_demo = configure_database_mode(mock_args)
            
            assert is_demo is True
            assert os.environ.get("USE_SQLITE") == "true"
            assert os.environ.get("SQLITE_PATH") == "custom.db"
    
    def test_configure_database_mode_postgresql(self):
        """Test configuring postgresql database mode (default)"""
        mock_args = MagicMock()
        mock_args.demo = False
        mock_args.sqlite = False
        
        with patch.dict(os.environ, {}, clear=True):
            is_demo = configure_database_mode(mock_args)
            
            assert is_demo is False
            assert os.environ.get("USE_SQLITE") != "true"


class TestApplicationConfiguration:
    """Test application configuration setup"""
    
    @patch('app.config.create_directories')
    @patch('app.utils.logging.setup_logging')
    @patch('app.config.get_settings')
    @patch('app.utils.logging.get_logger')
    def test_setup_application_config_basic(self, mock_get_logger, mock_get_settings,
                                          mock_setup_logging, mock_create_directories):
        """Test basic application configuration setup"""
        mock_settings = MagicMock()
        mock_settings.HOST = "0.0.0.0"
        mock_settings.PORT = 8000
        mock_settings.DEBUG = False
        mock_settings.RELOAD = False
        mock_get_settings.return_value = mock_settings
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_args = MagicMock()
        mock_args.host = None
        mock_args.port = None
        mock_args.debug = False
        mock_args.no_browser = False
        
        settings, logger = setup_application_config(mock_args, False)
        
        assert settings == mock_settings
        assert logger == mock_logger
        mock_setup_logging.assert_called_once()
        mock_create_directories.assert_called_once()
    
    @patch('app.config.create_directories')
    @patch('app.utils.logging.setup_logging')
    @patch('app.config.get_settings')
    @patch('app.utils.logging.get_logger')
    @patch('app.config.init_sqlite_database')
    def test_setup_application_config_demo_mode(self, mock_init_sqlite, mock_get_logger,
                                               mock_get_settings, mock_setup_logging,
                                               mock_create_directories):
        """Test application configuration setup in demo mode"""
        mock_settings = MagicMock()
        mock_settings.SQLITE_PATH = "demo.db"
        mock_get_settings.return_value = mock_settings
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_args = MagicMock()
        mock_args.host = None
        mock_args.port = None
        mock_args.debug = False
        mock_args.no_browser = False
        
        settings, logger = setup_application_config(mock_args, True)
        
        mock_init_sqlite.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('app.config.create_directories')
    @patch('app.utils.logging.setup_logging')
    @patch('app.config.get_settings')
    @patch('app.utils.logging.get_logger')
    def test_setup_application_config_command_line_overrides(self, mock_get_logger, mock_get_settings,
                                                            mock_setup_logging, mock_create_directories):
        """Test command line argument overrides"""
        mock_settings = MagicMock()
        mock_settings.HOST = "0.0.0.0"
        mock_settings.PORT = 8000
        mock_settings.DEBUG = False
        mock_settings.RELOAD = False
        mock_get_settings.return_value = mock_settings
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_args = MagicMock()
        mock_args.host = "127.0.0.1"
        mock_args.port = 9000
        mock_args.debug = True
        mock_args.no_browser = True
        
        settings, logger = setup_application_config(mock_args, False)
        
        assert settings.HOST == "127.0.0.1"
        assert settings.PORT == 9000
        assert settings.DEBUG is True
        assert settings.RELOAD is True
        assert settings.no_browser is True


class TestMainFunction:
    """Test main function"""
    
    @patch('uvicorn.run')
    @patch('app.main.create_app')
    @patch('app.main.setup_application_config')
    @patch('app.main.configure_database_mode')
    @patch('app.main.parse_arguments')
    def test_main_function_basic(self, mock_parse_args, mock_configure_db, mock_setup_config,
                               mock_create_app, mock_uvicorn_run):
        """Test main function execution"""
        mock_args = MagicMock()
        mock_args.demo = False
        mock_parse_args.return_value = mock_args
        
        mock_configure_db.return_value = False
        
        import tempfile
        import os
        
        # Create a temporary directory for static files
        temp_static_dir = tempfile.mkdtemp()
        
        mock_settings = MagicMock()
        mock_settings.HOST = "0.0.0.0"
        mock_settings.PORT = 8000
        mock_settings.RELOAD = False
        mock_settings.LOG_LEVEL = "INFO"
        mock_settings.APP_NAME = "EasyMLOps"
        mock_settings.APP_VERSION = "1.0.0"
        mock_settings.STATIC_DIR = temp_static_dir
        mock_settings.is_sqlite.return_value = False
        mock_settings.get_db_type.return_value = "postgresql"
        
        mock_logger = MagicMock()
        mock_setup_config.return_value = (mock_settings, mock_logger)
        
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        
        try:
            with patch('sys.modules') as mock_modules:
                mock_modules.__contains__ = lambda self, x: x != "pytest"
                
                from app.main import main
                main()
                
                mock_uvicorn_run.assert_called_once()
                call_args = mock_uvicorn_run.call_args
                assert call_args[1]['host'] == "0.0.0.0"
                assert call_args[1]['port'] == 8000
        finally:
            # Clean up temp directory
            if os.path.exists(temp_static_dir):
                try:
                    os.rmdir(temp_static_dir)
                except:
                    pass

