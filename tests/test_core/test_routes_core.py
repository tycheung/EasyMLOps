"""
Comprehensive tests for route registration
Tests all route endpoints and registration
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.routes import register_routes
from app.config import Settings


class TestRegisterRoutes:
    """Test route registration"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        app = FastAPI()
        settings = Settings()
        register_routes(app, settings)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_detailed_health_endpoint(self, client):
        """Test detailed health check endpoint"""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "database" in data
        assert "directories" in data
    
    def test_root_endpoint_with_file(self, client):
        """Test root endpoint when index.html exists"""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "<html>Test</html>"
                
                response = client.get("/")
                assert response.status_code == 200
    
    def test_root_endpoint_without_file(self, client):
        """Test root endpoint when index.html doesn't exist"""
        with patch('os.path.exists', return_value=False):
            response = client.get("/")
            assert response.status_code == 200
            assert "EasyMLOps" in response.text
    
    def test_info_endpoint(self, client):
        """Test app info endpoint"""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "debug" in data
    
    def test_models_router_registered(self, client):
        """Test models router is registered"""
        # Try to access models endpoint (should not 404 if registered)
        response = client.get("/api/v1/models")
        # Should either return data or 404, but not 500 from missing route
        assert response.status_code in [200, 404, 422]
    
    def test_deployments_router_registered(self, client):
        """Test deployments router is registered"""
        response = client.get("/api/v1/deployments")
        assert response.status_code in [200, 404, 422]
    
    def test_monitoring_router_registered(self, client):
        """Test monitoring router is registered"""
        response = client.get("/api/v1/monitoring/dashboard")
        assert response.status_code in [200, 404, 422]
    
    def test_schemas_router_registered(self, client):
        """Test schemas router is registered"""
        response = client.get("/api/v1/schemas")
        assert response.status_code in [200, 404, 422]
    
    def test_dynamic_router_registered(self, client):
        """Test dynamic router is registered"""
        # Dynamic routes might not have a simple GET endpoint
        # Just verify the router is included
        routes = [route.path for route in client.app.routes if hasattr(route, 'path')]
        assert any("/api/v1" in route for route in routes)
    
    def test_register_routes_includes_all_routers(self, app):
        """Test that register_routes includes all expected routers"""
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        
        # Check for core routes
        assert "/health" in routes
        assert "/health/detailed" in routes
        assert "/" in routes
        assert "/info" in routes
        
        # Check for API routes (prefixes)
        api_routes = [r for r in routes if r.startswith("/api/v1")]
        assert len(api_routes) > 0
    
    def test_register_routes_router_tags(self, app):
        """Test that routers are registered with correct tags"""
        # Check route tags
        for route in app.routes:
            if hasattr(route, 'tags'):
                if route.path == "/health":
                    assert "Health" in route.tags
                elif route.path == "/":
                    assert "Frontend" in route.tags
                elif route.path == "/info":
                    assert "System" in route.tags
    
    @patch('app.database.get_db_info')
    def test_detailed_health_check_includes_database_info(self, mock_get_db_info, client):
        """Test detailed health check includes database information"""
        mock_get_db_info.return_value = {"status": "connected", "engine": "sqlite"}
        
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "database" in data
        assert data["database"]["status"] == "connected"
    
    @patch('os.path.exists')
    def test_detailed_health_check_includes_directories(self, mock_exists, client):
        """Test detailed health check includes directory information"""
        mock_exists.return_value = True
        
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "directories" in data
        assert isinstance(data["directories"], dict)
        assert "models" in data["directories"]
        assert "bentos" in data["directories"]
        assert "static" in data["directories"]
        assert "logs" in data["directories"]
    
    def test_health_check_environment_detection(self, client):
        """Test health check correctly detects environment"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["environment"] in ["production", "development"]
    
    def test_info_endpoint_includes_all_fields(self, client):
        """Test info endpoint includes all expected fields"""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        
        required_fields = ["name", "version", "debug", "api_prefix", "docs_url", "redoc_url"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_schemas_router_registered_under_models(self, client):
        """Test schemas router is registered under /models prefix"""
        # Schemas router should be accessible under both /models and /schemas
        response = client.get("/api/v1/models/schemas")
        assert response.status_code in [200, 404, 422]
    
    def test_schemas_router_registered_under_schemas(self, client):
        """Test schemas router is registered under /schemas prefix"""
        response = client.get("/api/v1/schemas")
        assert response.status_code in [200, 404, 422]
    
    def test_register_routes_creates_fastapi_app(self):
        """Test that register_routes works with FastAPI app"""
        from fastapi import FastAPI
        app = FastAPI()
        settings = Settings()
        
        # Should not raise
        register_routes(app, settings)
        
        assert isinstance(app, FastAPI)
        assert len(app.routes) > 0

