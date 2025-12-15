"""
Tests for app/routes/dynamic.py module
Tests the re-export module structure
"""

import pytest
from app.routes.dynamic import router, DynamicRouteManager, route_manager


class TestDynamicRoutesModule:
    """Test dynamic routes module re-exports"""
    
    def test_router_imported(self):
        """Test router is imported from dynamic_routes"""
        assert router is not None
        assert hasattr(router, 'routes')
    
    def test_dynamic_route_manager_class_imported(self):
        """Test DynamicRouteManager class is imported"""
        assert DynamicRouteManager is not None
        assert isinstance(DynamicRouteManager, type)
    
    def test_route_manager_instance_imported(self):
        """Test route_manager instance is imported"""
        assert route_manager is not None
        # Should be an instance, not the class
        assert route_manager is not DynamicRouteManager
    
    def test_module_exports(self):
        """Test that module has expected exports"""
        import app.routes.dynamic as dynamic_module
        assert hasattr(dynamic_module, 'router')
        assert hasattr(dynamic_module, 'DynamicRouteManager')
        assert hasattr(dynamic_module, 'route_manager')

