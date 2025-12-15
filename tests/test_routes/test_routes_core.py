"""
Tests for core application routes
Tests health checks, info endpoints, and root endpoint
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from tests.conftest import get_test_app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(get_test_app())


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check_basic(self, client):
        """Test basic health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        result = response.json()
        assert "status" in result
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "version" in result
        assert "environment" in result
        assert "database_type" in result
        assert "mode" in result
    
    def test_health_check_detailed(self, client):
        """Test detailed health check endpoint"""
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        result = response.json()
        assert "status" in result
        assert result["status"] == "healthy"
        assert "database" in result
        assert "directories" in result
        assert isinstance(result["directories"], dict)
        assert "models" in result["directories"]
        assert "bentos" in result["directories"]
        assert "static" in result["directories"]
        assert "logs" in result["directories"]


class TestInfoEndpoint:
    """Test application info endpoint"""
    
    def test_app_info(self, client):
        """Test application info endpoint"""
        response = client.get("/info")
        
        assert response.status_code == 200
        result = response.json()
        assert "name" in result
        assert "version" in result
        assert "debug" in result
        assert "api_prefix" in result
        assert "docs_url" in result
        assert "redoc_url" in result
        assert "database_type" in result
        assert "mode" in result


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint_with_static_file(self, client):
        """Test root endpoint when static file exists"""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "<html>Test</html>"
                response = client.get("/")
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/html; charset=utf-8"
    
    def test_root_endpoint_fallback(self, client):
        """Test root endpoint fallback when static file doesn't exist"""
        with patch('os.path.exists', return_value=False):
            response = client.get("/")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/html; charset=utf-8"
            content = response.text
            assert "EasyMLOps" in content
            assert "Welcome" in content


class TestRouteRegistration:
    """Test route registration"""
    
    def test_routes_are_registered(self, client):
        """Test that routes are properly registered"""
        # Test that health endpoint exists
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test that info endpoint exists
        response = client.get("/info")
        assert response.status_code == 200
        
        # Test that root endpoint exists
        response = client.get("/")
        assert response.status_code == 200
    
    def test_api_docs_available(self, client):
        """Test that API documentation is available"""
        response = client.get("/docs")
        # Should return 200 or redirect
        assert response.status_code in [200, 307, 308]
        
        response = client.get("/redoc")
        # Should return 200 or redirect
        assert response.status_code in [200, 307, 308]

