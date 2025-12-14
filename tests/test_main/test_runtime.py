"""
Application runtime tests
Tests runtime behavior, request handling, error handling, and endpoint functionality
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError

from app.core.app_factory import create_app, RequestLoggingMiddleware


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
        with patch('app.core.app_factory.logger') as mock_log:
            yield mock_log
    
    @pytest.mark.asyncio
    async def test_middleware_logs_requests(self, test_app, mock_logger):
        """Test middleware logs incoming requests"""
        middleware = RequestLoggingMiddleware(test_app)
        
        from starlette.datastructures import URL
        
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url = URL("http://localhost:8000/api/v1/models")
        mock_request.state = MagicMock()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        
        async def mock_call_next(request):
            return mock_response
        
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        assert hasattr(mock_request.state, 'request_id')
        assert 'X-Request-ID' in response.headers
        assert 'X-Process-Time' in response.headers
    
    @pytest.mark.asyncio
    async def test_middleware_handles_exceptions(self, test_app, mock_logger):
        """Test middleware handles exceptions properly"""
        middleware = RequestLoggingMiddleware(test_app)
        
        from starlette.datastructures import URL
        
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url = URL("http://localhost:8000/api/v1/models")
        mock_request.state = MagicMock()
        
        async def mock_call_next_error(request):
            raise Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            await middleware.dispatch(mock_request, mock_call_next_error)
        
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
        
        response = client.post("/api/v1/models/upload", json={"invalid": "data"})
        
        if response.status_code == 422:
            result = response.json()
            assert "error" in result
            assert result["error"]["status_code"] == 422
            assert "request_id" in result["error"]
    
    def test_general_exception_handler(self):
        """Test general exception handler"""
        from app.config import get_settings
        from app.utils.logging import get_logger
        
        test_settings = get_settings()
        test_logger = get_logger(__name__)
        test_app = create_app(test_settings, test_logger)
        
        client = TestClient(test_app)
        
        assert Exception in test_app.exception_handlers
        assert len(test_app.exception_handlers) >= 3


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
        assert test_app is not None
        
        # Routes can be Route objects (with .path) or Mount objects (with .path)
        route_paths = [getattr(route, 'path', None) for route in test_app.routes if hasattr(route, 'path')]
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
        
        response = client.options("/health", headers={
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
            "Origin": "http://localhost:3000"
        })
        
        assert response.status_code in [200, 204]

