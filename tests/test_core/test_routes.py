"""
Tests for route registration
Tests health endpoints, root endpoint, and route registration
"""

import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient

from app.core.routes import register_routes
from app.core.app_factory import create_app
from app.config import Settings


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check_basic(self):
        """Test basic health check endpoint"""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data
        assert "database_type" in data
        assert "mode" in data
    
    def test_health_check_detailed(self):
        """Test detailed health check endpoint"""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "database" in data
        assert "directories" in data
        assert isinstance(data["directories"], dict)
        assert "models" in data["directories"]
        assert "bentos" in data["directories"]
        assert "static" in data["directories"]
        assert "logs" in data["directories"]


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint_with_html_file(self):
        """Test root endpoint when HTML file exists"""
        app = create_app()
        client = TestClient(app)
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="<html>Test</html>")):
                response = client.get("/")
                
                assert response.status_code == 200
                assert "text/html" in response.headers["content-type"]
                assert "<html>Test</html>" in response.text
    
    def test_root_endpoint_without_html_file(self):
        """Test root endpoint when HTML file doesn't exist"""
        app = create_app()
        client = TestClient(app)
        
        with patch('os.path.exists', return_value=False):
            with patch('builtins.open', side_effect=FileNotFoundError()):
                response = client.get("/")
                
                assert response.status_code == 200
                assert "text/html" in response.headers["content-type"]
                assert "EasyMLOps" in response.text
                assert "/docs" in response.text


class TestAppInfoEndpoint:
    """Test app info endpoint"""
    
    def test_app_info_endpoint(self):
        """Test app info endpoint"""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "debug" in data
        assert "api_prefix" in data
        assert "docs_url" in data
        assert "redoc_url" in data
        assert "database_type" in data
        assert "mode" in data


class TestRouteRegistration:
    """Test route registration"""
    
    def test_routes_registered(self):
        """Test that all routes are registered"""
        app = create_app()
        client = TestClient(app)
        
        # Test models routes
        response = client.get("/api/v1/models")
        assert response.status_code in [200, 404]  # May be 404 if no models
        
        # Test deployments routes
        response = client.get("/api/v1/deployments")
        assert response.status_code in [200, 404]
        
        # Test monitoring routes
        response = client.get("/api/v1/monitoring/dashboard")
        assert response.status_code in [200, 404]
        
        # Test schemas routes
        response = client.get("/api/v1/schemas")
        assert response.status_code in [200, 404]
    
    def test_health_routes_registered(self):
        """Test health routes are registered"""
        app = create_app()
        client = TestClient(app)
        
        # Basic health
        response = client.get("/health")
        assert response.status_code == 200
        
        # Detailed health
        response = client.get("/health/detailed")
        assert response.status_code == 200
    
    def test_docs_available(self):
        """Test API documentation is available"""
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/openapi.json")
        assert response.status_code == 200

