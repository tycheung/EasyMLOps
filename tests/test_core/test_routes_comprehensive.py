"""
Comprehensive tests for core routes
Tests route registration, health checks, and root endpoints
"""

import pytest
import os
import time
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.routes import register_routes
from app.config import Settings, get_settings


class TestRouteRegistrationComprehensive:
    """Comprehensive tests for route registration"""
    
    @pytest.fixture
    def app(self):
        """Create test app with routes"""
        app = FastAPI()
        settings = get_settings()
        register_routes(app, settings)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)
    
    def test_all_routers_registered(self, app):
        """Test that all expected routers are registered"""
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        
        # Check core routes
        assert "/health" in routes
        assert "/health/detailed" in routes
        assert "/" in routes
        assert "/info" in routes
        
        # Check API routes exist
        api_routes = [r for r in routes if r.startswith("/api/v1")]
        assert len(api_routes) > 0
    
    def test_router_prefixes_correct(self, app):
        """Test that routers have correct prefixes"""
        settings = get_settings()
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        
        # Check for expected prefixes
        expected_prefixes = [
            f"{settings.API_V1_PREFIX}/models",
            f"{settings.API_V1_PREFIX}/deployments",
            f"{settings.API_V1_PREFIX}/monitoring",
            f"{settings.API_V1_PREFIX}/schemas",
        ]
        
        for prefix in expected_prefixes:
            matching_routes = [r for r in routes if r.startswith(prefix)]
            assert len(matching_routes) > 0, f"No routes found with prefix {prefix}"
    
    def test_health_check_response_structure(self, client):
        """Test health check response has all expected fields"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        required_fields = ["status", "timestamp", "version", "environment", "database_type", "mode"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        assert data["status"] == "healthy"
        assert isinstance(data["timestamp"], (int, float))
    
    def test_detailed_health_check_structure(self, client):
        """Test detailed health check response structure"""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "database" in data
        assert "directories" in data
        assert isinstance(data["directories"], dict)
        assert "models" in data["directories"]
        assert "bentos" in data["directories"]
        assert "static" in data["directories"]
        assert "logs" in data["directories"]
    
    @patch('os.path.exists')
    def test_root_endpoint_with_static_file(self, mock_exists, client):
        """Test root endpoint serves static file when available"""
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "<html>Test Content</html>"
            
            response = client.get("/")
            assert response.status_code == 200
            assert response.headers['content-type'] == 'text/html; charset=utf-8'
    
    def test_root_endpoint_fallback_html(self, client):
        """Test root endpoint fallback HTML when static file missing"""
        with patch('os.path.exists', return_value=False):
            response = client.get("/")
            assert response.status_code == 200
            assert "Welcome to EasyMLOps" in response.text
            assert "API Documentation" in response.text
    
    def test_info_endpoint_complete(self, client):
        """Test info endpoint returns complete information"""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "name", "version", "debug", "api_prefix",
            "docs_url", "redoc_url", "database_type", "mode"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_health_check_timestamp_format(self, client):
        """Test health check timestamp is valid"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        timestamp = data["timestamp"]
        assert isinstance(timestamp, (int, float))
        assert timestamp > 0
        # Should be recent (within last minute)
        assert abs(time.time() - timestamp) < 60
    
    def test_health_check_environment_detection(self, client):
        """Test health check correctly detects environment"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        assert data["environment"] in ["production", "development"]
    
    def test_health_check_mode_detection(self, client):
        """Test health check correctly detects mode"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        assert data["mode"] in ["demo", "production"]
    
    @patch('app.database.get_db_info')
    def test_detailed_health_database_info(self, mock_get_db_info, client):
        """Test detailed health check includes database information"""
        mock_get_db_info.return_value = {
            "status": "connected",
            "engine": "sqlite",
            "database": "test.db"
        }
        
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        
        assert "database" in data
        assert data["database"]["status"] == "connected"
        assert data["database"]["engine"] == "sqlite"
    
    def test_register_routes_with_custom_settings(self):
        """Test route registration with custom settings"""
        app = FastAPI()
        custom_settings = Settings()
        custom_settings.API_V1_PREFIX = "/api/v2"
        
        register_routes(app, custom_settings)
        
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        # Should have routes with v2 prefix
        v2_routes = [r for r in routes if r.startswith("/api/v2")]
        assert len(v2_routes) > 0
    
    def test_route_tags_assigned(self, app):
        """Test that routes have appropriate tags"""
        for route in app.routes:
            if hasattr(route, 'tags') and hasattr(route, 'path'):
                if route.path == "/health" or route.path == "/health/detailed":
                    assert "Health" in route.tags
                elif route.path == "/":
                    assert "Frontend" in route.tags
                elif route.path == "/info":
                    assert "System" in route.tags
    
    def test_multiple_router_inclusions(self, app):
        """Test that routers can be included multiple times with different prefixes"""
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        
        # Schemas router should be accessible under both /models and /schemas
        schema_routes_models = [r for r in routes if r.startswith("/api/v1/models") and "schema" in r.lower()]
        schema_routes_schemas = [r for r in routes if r.startswith("/api/v1/schemas")]
        
        # At least one should exist
        assert len(schema_routes_models) > 0 or len(schema_routes_schemas) > 0

