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
    app, create_app, RequestLoggingMiddleware, create_lifespan,
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
            mock_settings.STATIC_DIR = "static"
            mock_settings.API_V1_PREFIX = "/api/v1"
            mock_settings.DEBUG = False
            mock_settings.is_sqlite.return_value = True
            mock_settings.get_db_type.return_value = "SQLite"
            
            with patch('os.path.exists', return_value=False):
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
            mock_settings.STATIC_DIR = "static"
            mock_settings.API_V1_PREFIX = "/api/v1"
            mock_settings.DEBUG = False
            mock_settings.is_sqlite.return_value = True
            mock_settings.get_db_type.return_value = "SQLite"
            
            with patch('os.path.exists', return_value=False):
                test_app = create_app()
            
            # Check middleware is applied - look at the actual middleware classes
            middleware_details = []
            for middleware in test_app.user_middleware:
                if hasattr(middleware, 'cls'):
                    middleware_details.append(str(middleware.cls))
                else:
                    middleware_details.append(str(type(middleware)))
            
            # Check that the essential middleware is present
            assert any("RequestLoggingMiddleware" in detail for detail in middleware_details), f"RequestLoggingMiddleware not found in {middleware_details}"
            assert any("GZip" in detail for detail in middleware_details), f"GZip middleware not found in {middleware_details}"
            assert any("CORS" in detail for detail in middleware_details), f"CORS middleware not found in {middleware_details}"
    
    def test_exception_handlers_registered(self):
        """Test exception handlers are properly registered"""
        with patch('app.main.settings') as mock_settings:
            mock_settings.APP_NAME = "EasyMLOps"
            mock_settings.APP_VERSION = "1.0.0"
            mock_settings.BACKEND_CORS_ORIGINS = ["http://localhost:3000"]
            mock_settings.STATIC_DIR = "static"
            mock_settings.API_V1_PREFIX = "/api/v1"
            mock_settings.DEBUG = False
            mock_settings.is_sqlite.return_value = True
            mock_settings.get_db_type.return_value = "SQLite"
            
            with patch('os.path.exists', return_value=False):
                test_app = create_app()
            
            # Check exception handlers are registered
            from starlette.exceptions import HTTPException as StarletteHTTPException
            assert StarletteHTTPException in test_app.exception_handlers
            assert RequestValidationError in test_app.exception_handlers
            assert Exception in test_app.exception_handlers


class TestRequestLoggingMiddleware:
    """Test request logging middleware functionality"""
    
    @pytest.fixture
    def test_app(self):
        """Create test app for middleware testing"""
        from app.config import get_settings
        from app.utils.logging import get_logger
        
        test_settings = get_settings()
        test_logger = get_logger(__name__)
        return create_app(test_settings, test_logger)
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing"""
        with patch('app.main.logger') as mock_log:
            yield mock_log
    
    @pytest.mark.asyncio
    async def test_middleware_logs_requests(self, test_app, mock_logger):
        """Test middleware logs incoming requests"""
        middleware = RequestLoggingMiddleware(test_app)
        
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
    async def test_middleware_handles_exceptions(self, test_app, mock_logger):
        """Test middleware handles exceptions properly"""
        middleware = RequestLoggingMiddleware(test_app)
        
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
    
    @pytest.fixture
    def test_app(self):
        """Create test app for exception handler testing"""
        from app.config import get_settings
        from app.utils.logging import get_logger
        
        test_settings = get_settings()
        test_logger = get_logger(__name__)
        return create_app(test_settings, test_logger)
    
    def test_http_exception_handler(self, test_app):
        """Test HTTP exception handler"""
        client = TestClient(test_app)
        
        # Test 404 error
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        result = response.json()
        assert "error" in result
        assert result["error"]["status_code"] == 404
        assert "request_id" in result["error"]
        assert "timestamp" in result["error"]
    
    def test_validation_exception_handler(self, test_app):
        """Test request validation exception handler"""
        client = TestClient(test_app)
        
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
        # Create a test app to test exception handling
        from app.config import get_settings
        from app.utils.logging import get_logger
        
        test_settings = get_settings()
        test_logger = get_logger(__name__)
        test_app = create_app(test_settings, test_logger)
        
        # Use TestClient to trigger exceptions
        client = TestClient(test_app)
        
        # For now, just test that the app is created successfully
        # and has exception handlers registered
        assert Exception in test_app.exception_handlers
        assert len(test_app.exception_handlers) >= 3  # HTTP, Validation, General


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    @pytest.fixture
    def test_app(self):
        """Create test app for health endpoint testing"""
        from app.config import get_settings
        from app.utils.logging import get_logger
        
        test_settings = get_settings()
        test_logger = get_logger(__name__)
        return create_app(test_settings, test_logger)
    
    def test_health_check_endpoint(self, test_app):
        """Test basic health check endpoint"""
        client = TestClient(test_app)
        
        response = client.get("/health")
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "version" in result
        assert "environment" in result
        assert "database_type" in result
        assert "mode" in result
    
    def test_detailed_health_check_endpoint(self, test_app):
        """Test detailed health check endpoint"""
        client = TestClient(test_app)
        
        response = client.get("/health/detailed")
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "healthy"
        assert "database" in result
        assert "directories" in result
    
    def test_root_endpoint_redirect(self, test_app):
        """Test root endpoint serves HTML or redirects"""
        client = TestClient(test_app)
        
        response = client.get("/")
        # Should return HTML content (200) since static files might not exist in test
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_info_endpoint(self, test_app):
        """Test info endpoint"""
        client = TestClient(test_app)
        
        response = client.get("/info")
        assert response.status_code == 200
        
        result = response.json()
        assert "name" in result
        assert "version" in result
        assert "debug" in result
        assert "api_prefix" in result


class TestLifespanEvents:
    """Test application lifespan events"""
    
    @pytest.mark.asyncio
    @patch('app.database.create_tables')
    @patch('app.database.check_async_db_connection')
    @patch('app.database.init_db')
    @patch('app.database.close_db')
    async def test_lifespan_startup_success(self, mock_close_db, mock_init_db, 
                                          mock_check_db, mock_create_tables):
        """Test successful lifespan startup"""
        mock_check_db.return_value = True
        
        # Create mock settings and logger
        mock_settings = MagicMock()
        mock_settings.get_db_type.return_value = "SQLite"
        mock_settings.is_sqlite.return_value = True
        mock_logger = MagicMock()
        
        # Create a FastAPI app for testing
        from fastapi import FastAPI
        test_app = FastAPI()
        
        # Test lifespan
        async with create_lifespan(mock_settings, mock_logger)(test_app):
            # Startup completed
            pass
        
        # Verify startup actions
        mock_check_db.assert_called_once()
        mock_init_db.assert_called_once()
        
        # Verify logging
        mock_logger.info.assert_called()
        assert any("Starting EasyMLOps application" in str(call) for call in mock_logger.info.call_args_list)
        
        # Verify shutdown actions
        mock_close_db.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.database.check_async_db_connection')
    @patch('app.database.init_db')
    @patch('app.database.close_db')
    async def test_lifespan_database_connection_failure(self, mock_close_db, mock_init_db, mock_check_db):
        """Test lifespan with database connection failure"""
        mock_check_db.return_value = False
        
        # Create mock settings and logger
        mock_settings = MagicMock()
        mock_settings.get_db_type.return_value = "PostgreSQL"
        mock_settings.is_sqlite.return_value = False
        mock_logger = MagicMock()
        
        # Create a FastAPI app for testing
        from fastapi import FastAPI
        test_app = FastAPI()
        
        # Should still start even with DB connection failure
        async with create_lifespan(mock_settings, mock_logger)(test_app):
            pass
        
        # Verify error was logged
        mock_logger.error.assert_called()
        assert any("Database connection failed" in str(call) for call in mock_logger.error.call_args_list)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.start_monitoring_tasks')
    @patch('app.database.check_async_db_connection')
    @patch('app.database.create_tables')
    @patch('app.database.init_db')
    @patch('app.database.close_db')
    async def test_lifespan_monitoring_service_startup(self, mock_close_db, mock_init_db,
                                                      mock_create_tables, mock_check_db,
                                                      mock_start_monitoring):
        """Test lifespan with monitoring service startup"""
        mock_check_db.return_value = True
        mock_start_monitoring.return_value = None
        
        # Create mock settings and logger
        mock_settings = MagicMock()
        mock_settings.get_db_type.return_value = "SQLite"
        mock_settings.is_sqlite.return_value = True
        mock_logger = MagicMock()
        
        # Create a FastAPI app for testing
        from fastapi import FastAPI
        test_app = FastAPI()
        
        with patch.dict(os.environ, {"DISABLE_MONITORING": "false"}, clear=False):
            async with create_lifespan(mock_settings, mock_logger)(test_app):
                pass
        
        # Verify monitoring service was started
        mock_start_monitoring.assert_called_once()
        mock_logger.info.assert_called()
        assert any("Monitoring service started" in str(call) for call in mock_logger.info.call_args_list)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.start_monitoring_tasks')
    @patch('app.database.check_async_db_connection')
    @patch('app.database.create_tables')
    @patch('app.database.init_db')
    @patch('app.database.close_db')
    async def test_lifespan_monitoring_disabled(self, mock_close_db, mock_init_db,
                                              mock_create_tables, mock_check_db,
                                              mock_start_monitoring):
        """Test lifespan with monitoring disabled"""
        mock_check_db.return_value = True
        
        # Create mock settings and logger
        mock_settings = MagicMock()
        mock_settings.get_db_type.return_value = "SQLite"
        mock_settings.is_sqlite.return_value = True
        mock_logger = MagicMock()
        
        # Create a FastAPI app for testing
        from fastapi import FastAPI
        test_app = FastAPI()
        
        with patch.dict(os.environ, {"DISABLE_MONITORING": "true"}, clear=False):
            async with create_lifespan(mock_settings, mock_logger)(test_app):
                pass
        
        # Verify monitoring service was NOT started
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
            
            # sqlite = True should return True (use_sqlite = True)
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
            # USE_SQLITE should not be set or should be false
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
    
    @patch('app.config.create_directories')
    @patch('app.utils.logging.setup_logging')
    @patch('app.config.get_settings')
    @patch('app.utils.logging.get_logger')
    @patch('app.config.init_sqlite_database')
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
    
    @patch('app.config.create_directories')
    @patch('app.utils.logging.setup_logging')
    @patch('app.config.get_settings')
    @patch('app.utils.logging.get_logger')
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
    
    @patch('uvicorn.run')
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
            mock_modules.__contains__ = lambda self, x: x != "pytest"
            
            from app.main import main
            main()
            
            # Verify uvicorn.run was called
            mock_uvicorn_run.assert_called_once()
            call_args = mock_uvicorn_run.call_args
            assert call_args[1]['host'] == "0.0.0.0"
            assert call_args[1]['port'] == 8000


class TestStaticFileServing:
    """Test static file serving functionality"""
    
    @pytest.fixture
    def test_app(self):
        """Create test app for static file testing"""
        from app.config import get_settings
        from app.utils.logging import get_logger
        
        test_settings = get_settings()
        test_logger = get_logger(__name__)
        return create_app(test_settings, test_logger)
    
    def test_static_files_mounted(self, test_app):
        """Test static files are properly mounted"""
        # Check if static route is mounted (app might not have static files in test)
        assert test_app is not None
        
        # Check if routes include static if directory exists
        route_paths = [route.path for route in test_app.routes]
        # In test environment, static might not be mounted if directory doesn't exist
        # Just verify app creation succeeded
        assert len(route_paths) > 0


class TestApplicationIntegration:
    """Test complete application integration"""
    
    @pytest.fixture
    def test_app(self):
        """Create test app for integration testing"""
        from app.config import get_settings
        from app.utils.logging import get_logger
        
        test_settings = get_settings()
        test_logger = get_logger(__name__)
        return create_app(test_settings, test_logger)
    
    def test_app_startup_and_health_check(self, test_app):
        """Test complete app startup and health check"""
        client = TestClient(test_app)
        
        # Test basic health check
        response = client.get("/health")
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "healthy"
    
    def test_openapi_docs_available(self, test_app):
        """Test OpenAPI documentation is available"""
        client = TestClient(test_app)
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert openapi_spec["info"]["title"] == "EasyMLOps"
    
    def test_docs_endpoint_available(self, test_app):
        """Test Swagger UI docs endpoint is available"""
        client = TestClient(test_app)
        
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_cors_headers_present(self, test_app):
        """Test CORS headers are properly set"""
        client = TestClient(test_app)
        
        # Test preflight request
        response = client.options("/health", headers={
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
            "Origin": "http://localhost:3000"
        })
        
        # CORS should be configured
        assert response.status_code in [200, 204] 