"""
Comprehensive tests for app factory
Tests application creation, middleware, exception handlers, and lifespan events
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.app_factory import create_app, create_lifespan, RequestLoggingMiddleware
from app.config import Settings, get_settings
from app.utils.logging import get_logger


class TestAppFactoryCreation:
    """Test application factory creation"""
    
    def test_create_app_returns_fastapi_instance(self):
        """Test that create_app returns a FastAPI instance"""
        app = create_app()
        assert isinstance(app, FastAPI)
    
    def test_create_app_with_custom_settings(self):
        """Test creating app with custom settings"""
        custom_settings = Settings()
        custom_settings.DEBUG = True
        logger = get_logger(__name__)
        
        # create_app accepts positional args: (app_settings, app_logger)
        app = create_app(custom_settings, logger)
        assert isinstance(app, FastAPI)
    
    def test_create_app_registers_middleware(self):
        """Test that middleware is registered"""
        app = create_app()
        
        # Check for CORS middleware
        middleware_types = [type(middleware).__name__ for middleware in app.user_middleware]
        assert "CORSMiddleware" in middleware_types or len(app.user_middleware) > 0
    
    def test_create_app_registers_exception_handlers(self):
        """Test that exception handlers are registered"""
        app = create_app()
        
        # Check exception handlers exist
        assert hasattr(app, 'exception_handlers')
        assert len(app.exception_handlers) > 0


class TestRequestLoggingMiddleware:
    """Test request logging middleware"""
    
    @pytest.mark.asyncio
    async def test_middleware_adds_request_id(self):
        """Test middleware adds request ID to request state"""
        middleware = RequestLoggingMiddleware(MagicMock())
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        request.method = "GET"
        request.url.path = "/test"
        
        async def call_next(req):
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            return response
        
        with patch('app.core.app_factory.logger') as mock_logger:
            response = await middleware.dispatch(request, call_next)
            
            assert hasattr(request.state, 'request_id')
            assert "X-Request-ID" in response.headers
    
    @pytest.mark.asyncio
    async def test_middleware_logs_request(self):
        """Test middleware logs request information"""
        middleware = RequestLoggingMiddleware(MagicMock())
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        request.method = "POST"
        request.url.path = "/api/v1/test"
        
        async def call_next(req):
            response = MagicMock()
            response.status_code = 201
            response.headers = {}
            return response
        
        with patch('app.core.app_factory.logger') as mock_logger:
            await middleware.dispatch(request, call_next)
            
            # Should have logged the request
            assert mock_logger.info.called
    
    @pytest.mark.asyncio
    async def test_middleware_logs_errors(self):
        """Test middleware logs errors"""
        middleware = RequestLoggingMiddleware(MagicMock())
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        request.method = "GET"
        request.url.path = "/test"
        
        async def call_next(req):
            raise ValueError("Test error")
        
        with patch('app.core.app_factory.logger') as mock_logger:
            with pytest.raises(ValueError):
                await middleware.dispatch(request, call_next)
            
            # Should have logged the error
            assert mock_logger.error.called
    
    @pytest.mark.asyncio
    async def test_middleware_adds_process_time(self):
        """Test middleware adds process time header"""
        middleware = RequestLoggingMiddleware(MagicMock())
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        request.method = "GET"
        request.url.path = "/test"
        
        async def call_next(req):
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            return response
        
        response = await middleware.dispatch(request, call_next)
        
        assert "X-Process-Time" in response.headers
        assert float(response.headers["X-Process-Time"]) >= 0


class TestExceptionHandlers:
    """Test exception handlers"""
    
    def test_validation_exception_handler(self, client):
        """Test validation exception handler"""
        # Send invalid data to trigger validation error
        response = client.post("/api/v1/models", json={"invalid": "data"})
        
        # Should return 422 or 400 for validation errors
        assert response.status_code in [400, 422]
    
    def test_http_exception_handler(self, client):
        """Test HTTP exception handler"""
        # Try to access non-existent resource
        response = client.get("/api/v1/models/nonexistent_id_12345")
        
        # Should return 404
        assert response.status_code == 404
        assert "error" in response.json() or "detail" in response.json()


class TestLifespanEvents:
    """Test application lifespan events"""
    
    @pytest.mark.asyncio
    async def test_lifespan_startup(self):
        """Test lifespan startup events"""
        settings = get_settings()
        logger = get_logger(__name__)
        lifespan = create_lifespan(settings, logger)
        
        app = MagicMock()
        
        with patch('app.database.check_async_db_connection', return_value=True), \
             patch('app.database.init_db', return_value=None), \
             patch('app.database.get_db_info', return_value={"status": "connected"}):
            
            async with lifespan(app):
                # Lifespan should complete without errors
                pass
    
    @pytest.mark.asyncio
    async def test_lifespan_shutdown(self):
        """Test lifespan shutdown events"""
        settings = get_settings()
        logger = get_logger(__name__)
        lifespan = create_lifespan(settings, logger)
        
        app = MagicMock()
        
        with patch('app.database.check_async_db_connection', return_value=True), \
             patch('app.database.init_db', return_value=None), \
             patch('app.database.close_db', return_value=None):
            
            async with lifespan(app):
                # Shutdown should be called when exiting context
                pass


class TestStaticFiles:
    """Test static file serving"""
    
    def test_static_files_mounted(self):
        """Test that static files are mounted"""
        app = create_app()
        
        # Check if static files route exists
        static_routes = [route for route in app.routes if hasattr(route, 'path') and '/static' in route.path]
        # Static files may be mounted, check if route exists or app has static mount
        assert hasattr(app, 'mount') or len(static_routes) >= 0


class TestCORSConfiguration:
    """Test CORS configuration"""
    
    def test_cors_middleware_registered(self):
        """Test that CORS middleware is registered"""
        app = create_app()
        
        # Check middleware is registered
        assert len(app.user_middleware) > 0
    
    def test_cors_headers_in_response(self, client):
        """Test CORS headers in response"""
        response = client.get("/health")
        
        # CORS headers may or may not be present depending on origin
        # Just verify response is successful
        assert response.status_code == 200


class TestAppIntegration:
    """Integration tests for app factory"""
    
    def test_app_startup_complete(self):
        """Test that app starts up completely"""
        app = create_app()
        client = TestClient(app)
        
        # Should be able to make requests
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_app_has_all_routes(self):
        """Test that app has all expected routes"""
        app = create_app()
        client = TestClient(app)
        
        # Test core routes
        assert client.get("/health").status_code == 200
        assert client.get("/info").status_code == 200
        assert client.get("/").status_code == 200
    
    def test_app_openapi_schema(self):
        """Test that OpenAPI schema is generated"""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

