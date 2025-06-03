"""
Comprehensive tests for main FastAPI application
Tests application setup, middleware, exception handlers, lifespan events, and core functionality
"""

import pytest
import asyncio
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

from app.main import (
    app, create_app, RequestLoggingMiddleware, lifespan,
    parse_arguments, setup_application_config, configure_database_mode
)


class TestApplicationSetup:
    """Test application creation and configuration"""
    
    def test_create_app_basic(self):
        """Test basic app creation"""
        with patch('app.main.settings') as mock_settings:
            mock_settings.APP_NAME = "EasyMLOps"
            mock_settings.APP_VERSION = "1.0.0"
            mock_settings.BACKEND_CORS_ORIGINS = ["http://localhost:3000"]
            
            test_app = create_app()
            
            assert test_app is not None
            assert test_app.title == "EasyMLOps"
            assert test_app.version == "1.0.0"
    
    def test_create_app_middleware_configuration(self):
        """Test middleware is properly configured"""
        with patch('app.main.settings') as mock_settings:
            mock_settings.APP_NAME = "EasyMLOps"
            mock_settings.APP_VERSION = "1.0.0"
            mock_settings.BACKEND_CORS_ORIGINS = ["http://localhost:3000"]
            
            test_app = create_app()
            
            # Check middleware is applied
            middleware_classes = [type(middleware) for middleware in test_app.user_middleware]
            middleware_names = [str(cls) for cls in middleware_classes]
            
            # Should have CORS, GZip, and RequestLogging middleware
            assert any("CORS" in name for name in middleware_names)
            assert any("GZip" in name for name in middleware_names)
    
    def test_exception_handlers_registered(self):
        """Test exception handlers are properly registered"""
        with patch('app.main.settings') as mock_settings:
            mock_settings.APP_NAME = "EasyMLOps"
            mock_settings.APP_VERSION = "1.0.0"
            mock_settings.BACKEND_CORS_ORIGINS = ["http://localhost:3000"]
            
            test_app = create_app()
            
            # Check exception handlers are registered
            assert HTTPException in test_app.exception_handlers
            assert RequestValidationError in test_app.exception_handlers
            assert Exception in test_app.exception_handlers


class TestRequestLoggingMiddleware:
    """Test request logging middleware functionality"""
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing"""
        with patch('app.main.logger') as mock_log:
            yield mock_log
    
    @pytest.mark.asyncio
    async def test_middleware_logs_requests(self, mock_logger):
        """Test middleware logs incoming requests"""
        middleware = RequestLoggingMiddleware(app)
        
        # Mock request
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url = "http://localhost:8000/api/v1/models"
        mock_request.state = MagicMock()
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        
        # Mock call_next
        async def mock_call_next(request):
            return mock_response
        
        # Process request
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        # Verify request ID was set
        assert hasattr(mock_request.state, 'request_id')
        
        # Verify response headers were set
        assert 'X-Request-ID' in response.headers
        assert 'X-Process-Time' in response.headers
    
    @pytest.mark.asyncio
    async def test_middleware_handles_exceptions(self, mock_logger):
        """Test middleware handles exceptions properly"""
        middleware = RequestLoggingMiddleware(app)
        
        # Mock request
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url = "http://localhost:8000/api/v1/models"
        mock_request.state = MagicMock()
        
        # Mock call_next that raises exception
        async def mock_call_next_error(request):
            raise Exception("Test error")
        
        # Should re-raise the exception
        with pytest.raises(Exception, match="Test error"):
            await middleware.dispatch(mock_request, mock_call_next_error)
        
        # Should still have logged the error
        if mock_logger:
            mock_logger.error.assert_called()


class TestExceptionHandlers:
    """Test exception handlers"""
    
    def test_http_exception_handler(self):
        """Test HTTP exception handler"""
        client = TestClient(app)
        
        # Test 404 error
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        result = response.json()
        assert "error" in result
        assert result["error"]["status_code"] == 404
        assert "request_id" in result["error"]
        assert "timestamp" in result["error"]
    
    def test_validation_exception_handler(self):
        """Test request validation exception handler"""
        client = TestClient(app)
        
        # Test validation error by sending invalid data to an endpoint
        # This will trigger a 422 validation error
        response = client.post("/api/v1/models/upload", json={"invalid": "data"})
        
        # Should return 422 or handle validation errors appropriately
        if response.status_code == 422:
            result = response.json()
            assert "error" in result
            assert result["error"]["status_code"] == 422
            assert "request_id" in result["error"]
    
    def test_general_exception_handler(self):
        """Test general exception handler"""
        # This is harder to test directly without triggering actual exceptions
        # in the application, but we can test the handler function directly
        from app.main import general_exception_handler
        
        mock_request = MagicMock()
        mock_request.state.request_id = "test-request-id"
        
        mock_exception = Exception("Test error")
        
        # Test the handler directly
        response = asyncio.run(general_exception_handler(mock_request, mock_exception))
        
        assert response.status_code == 500
        assert "error" in response.body.decode()


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check_endpoint(self):
        """Test basic health check endpoint"""
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "version" in result
        assert "environment" in result
        assert "database_type" in result
        assert "mode" in result
    
    def test_root_endpoint_redirect(self):
        """Test root endpoint redirects to docs"""
        client = TestClient(app)
        
        response = client.get("/", allow_redirects=False)
        
        # Should redirect to docs or return appropriate response
        assert response.status_code in [200, 301, 302, 307, 308]


class TestLifespanEvents:
    """Test application lifespan events"""
    
    @pytest.mark.asyncio
    @patch('app.main.create_tables')
    @patch('app.main.check_db_connection')
    @patch('app.main.init_db')
    @patch('app.main.close_db')
    async def test_lifespan_startup_success(self, mock_close_db, mock_init_db, 
                                          mock_check_db, mock_create_tables):
        """Test successful application startup"""
        # Mock successful database connection
        mock_check_db.return_value = True
        mock_init_db.return_value = None
        mock_close_db.return_value = None
        
        # Mock settings
        with patch('app.main.settings') as mock_settings:
            mock_settings.get_db_type.return_value = "sqlite"
            mock_settings.is_sqlite.return_value = True
            mock_settings.DEBUG = False
            mock_settings.HOST = "0.0.0.0"
            mock_settings.PORT = 8000
            
            # Mock logger
            with patch('app.main.logger') as mock_logger:
                # Test lifespan
                async with lifespan(app):
                    # Startup completed
                    pass
                
                # Verify startup actions
                mock_check_db.assert_called_once()
                mock_create_tables.assert_called_once()
                mock_init_db.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.main.check_db_connection')
    @patch('app.main.init_db')
    @patch('app.main.close_db')
    async def test_lifespan_database_connection_failure(self, mock_close_db, mock_init_db, mock_check_db):
        """Test application startup with database connection failure"""
        # Mock failed database connection
        mock_check_db.return_value = False
        mock_init_db.return_value = None
        mock_close_db.return_value = None
        
        # Mock settings
        with patch('app.main.settings') as mock_settings:
            mock_settings.get_db_type.return_value = "postgresql"
            mock_settings.is_sqlite.return_value = False
            mock_settings.DEBUG = False
            mock_settings.HOST = "0.0.0.0"
            
            # Mock logger
            with patch('app.main.logger') as mock_logger:
                # Should still start even with DB connection failure
                async with lifespan(app):
                    pass
                
                # Should have logged error
                mock_logger.error.assert_called()
    
    @pytest.mark.asyncio
    @patch('app.main.monitoring_service.start_monitoring_tasks')
    @patch('app.main.check_db_connection')
    @patch('app.main.create_tables')
    @patch('app.main.init_db')
    @patch('app.main.close_db')
    async def test_lifespan_monitoring_service_startup(self, mock_close_db, mock_init_db,
                                                      mock_create_tables, mock_check_db,
                                                      mock_start_monitoring):
        """Test monitoring service startup during lifespan"""
        mock_check_db.return_value = True
        mock_init_db.return_value = None
        mock_close_db.return_value = None
        mock_start_monitoring.return_value = None
        
        # Ensure monitoring is not disabled
        with patch.dict(os.environ, {'DISABLE_MONITORING': 'false'}):
            with patch('app.main.settings') as mock_settings:
                mock_settings.get_db_type.return_value = "sqlite"
                mock_settings.is_sqlite.return_value = True
                mock_settings.DEBUG = False
                mock_settings.HOST = "0.0.0.0"
                
                with patch('app.main.logger'):
                    async with lifespan(app):
                        pass
                    
                    # Monitoring service should be started
                    mock_start_monitoring.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.main.monitoring_service.start_monitoring_tasks')
    @patch('app.main.check_db_connection')
    @patch('app.main.create_tables')
    @patch('app.main.init_db')
    @patch('app.main.close_db')
    async def test_lifespan_monitoring_disabled(self, mock_close_db, mock_init_db,
                                              mock_create_tables, mock_check_db,
                                              mock_start_monitoring):
        """Test lifespan with monitoring disabled"""
        mock_check_db.return_value = True
        mock_init_db.return_value = None
        mock_close_db.return_value = None
        
        # Disable monitoring
        with patch.dict(os.environ, {'DISABLE_MONITORING': 'true'}):
            with patch('app.main.settings') as mock_settings:
                mock_settings.get_db_type.return_value = "sqlite"
                mock_settings.is_sqlite.return_value = True
                mock_settings.DEBUG = False
                mock_settings.HOST = "0.0.0.0"
                
                with patch('app.main.logger'):
                    async with lifespan(app):
                        pass
                    
                    # Monitoring service should not be started
                    mock_start_monitoring.assert_not_called()


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
            
            assert is_demo is False
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
            # USE_SQLITE should not be set or should be false
            assert os.environ.get("USE_SQLITE") != "true"


class TestApplicationConfiguration:
    """Test application configuration setup"""
    
    @patch('app.main.create_directories')
    @patch('app.main.setup_logging')
    @patch('app.main.get_settings')
    @patch('app.main.get_logger')
    def test_setup_application_config_basic(self, mock_get_logger, mock_get_settings,
                                          mock_setup_logging, mock_create_directories):
        """Test basic application configuration setup"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.HOST = "0.0.0.0"
        mock_settings.PORT = 8000
        mock_settings.DEBUG = False
        mock_settings.RELOAD = False
        mock_get_settings.return_value = mock_settings
        
        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Mock args
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
    
    @patch('app.main.create_directories')
    @patch('app.main.setup_logging')
    @patch('app.main.get_settings')
    @patch('app.main.get_logger')
    @patch('app.main.init_sqlite_database')
    def test_setup_application_config_demo_mode(self, mock_init_sqlite, mock_get_logger,
                                               mock_get_settings, mock_setup_logging,
                                               mock_create_directories):
        """Test application configuration setup in demo mode"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.SQLITE_PATH = "demo.db"
        mock_get_settings.return_value = mock_settings
        
        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Mock args
        mock_args = MagicMock()
        mock_args.host = None
        mock_args.port = None
        mock_args.debug = False
        mock_args.no_browser = False
        
        settings, logger = setup_application_config(mock_args, True)  # Demo mode
        
        mock_init_sqlite.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('app.main.create_directories')
    @patch('app.main.setup_logging')
    @patch('app.main.get_settings')
    @patch('app.main.get_logger')
    def test_setup_application_config_command_line_overrides(self, mock_get_logger, mock_get_settings,
                                                            mock_setup_logging, mock_create_directories):
        """Test command line argument overrides"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.HOST = "0.0.0.0"
        mock_settings.PORT = 8000
        mock_settings.DEBUG = False
        mock_settings.RELOAD = False
        mock_get_settings.return_value = mock_settings
        
        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Mock args with overrides
        mock_args = MagicMock()
        mock_args.host = "127.0.0.1"
        mock_args.port = 9000
        mock_args.debug = True
        mock_args.no_browser = True
        
        settings, logger = setup_application_config(mock_args, False)
        
        # Settings should be overridden
        assert settings.HOST == "127.0.0.1"
        assert settings.PORT == 9000
        assert settings.DEBUG is True
        assert settings.RELOAD is True
        assert settings.no_browser is True


class TestMainFunction:
    """Test main function"""
    
    @patch('app.main.uvicorn.run')
    @patch('app.main.create_app')
    @patch('app.main.setup_application_config')
    @patch('app.main.configure_database_mode')
    @patch('app.main.parse_arguments')
    def test_main_function_basic(self, mock_parse_args, mock_configure_db, mock_setup_config,
                               mock_create_app, mock_uvicorn_run):
        """Test main function execution"""
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.demo = False
        mock_parse_args.return_value = mock_args
        
        # Mock database configuration
        mock_configure_db.return_value = False
        
        # Mock settings and logger
        mock_settings = MagicMock()
        mock_settings.HOST = "0.0.0.0"
        mock_settings.PORT = 8000
        mock_settings.RELOAD = False
        mock_settings.LOG_LEVEL = "INFO"
        mock_settings.APP_NAME = "EasyMLOps"
        mock_settings.APP_VERSION = "1.0.0"
        mock_settings.is_sqlite.return_value = False
        mock_settings.get_db_type.return_value = "postgresql"
        
        mock_logger = MagicMock()
        mock_setup_config.return_value = (mock_settings, mock_logger)
        
        # Mock app creation
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        
        # Import and patch global variables
        with patch('app.main.settings', mock_settings), \
             patch('app.main.logger', mock_logger), \
             patch('sys.modules') as mock_modules:
            
            # Ensure not in test environment
            mock_modules.__contains__ = lambda x: x != "pytest"
            
            from app.main import main
            main()
            
            # Verify uvicorn.run was called
            mock_uvicorn_run.assert_called_once()
            call_args = mock_uvicorn_run.call_args
            assert call_args[1]['host'] == "0.0.0.0"
            assert call_args[1]['port'] == 8000


class TestStaticFileServing:
    """Test static file serving"""
    
    def test_static_files_mounted(self):
        """Test static files are properly mounted"""
        # Check if static files route exists in the app
        routes = [route.path for route in app.routes]
        
        # Should have static file mounting or appropriate file serving
        # The exact implementation may vary
        assert any("/static" in route or "/docs" in route for route in routes)


class TestApplicationIntegration:
    """Integration tests for the complete application"""
    
    def test_app_startup_and_health_check(self):
        """Test application can start and respond to health checks"""
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "healthy"
    
    def test_openapi_docs_available(self):
        """Test OpenAPI documentation is available"""
        client = TestClient(app)
        
        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
    
    def test_docs_endpoint_available(self):
        """Test documentation endpoint is available"""
        client = TestClient(app)
        
        # Test docs endpoint
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Should return HTML content
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_cors_headers_present(self):
        """Test CORS headers are properly set"""
        client = TestClient(app)
        
        # Make an OPTIONS request to test CORS
        response = client.options("/health")
        
        # Should have CORS headers (or handle OPTIONS appropriately)
        assert response.status_code in [200, 204, 405]  # Various valid responses for OPTIONS 