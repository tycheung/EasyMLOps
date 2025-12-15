"""
Tests for dynamic routes module
Tests re-exports and module structure
"""

import pytest
from app.routes.dynamic import router, DynamicRouteManager, route_manager


class TestDynamicModule:
    """Test dynamic routes module exports"""
    
    def test_router_exported(self):
        """Test that router is exported"""
        assert router is not None
        assert hasattr(router, 'routes')
    
    def test_dynamic_route_manager_exported(self):
        """Test that DynamicRouteManager class is exported"""
        assert DynamicRouteManager is not None
        assert isinstance(DynamicRouteManager, type)
    
    def test_route_manager_instance_exported(self):
        """Test that route_manager instance is exported"""
        assert route_manager is not None
        # route_manager should be an instance - check it's callable or has attributes
        # The instance is created at module level, so it should exist
        assert route_manager is not DynamicRouteManager  # Should be instance, not class
    
    def test_route_manager_is_singleton(self):
        """Test that route_manager is a singleton instance"""
        from app.routes.dynamic import route_manager as route_manager2
        assert route_manager is route_manager2
    
    def test_router_has_routes(self):
        """Test that router has registered routes"""
        # Router should have at least some routes
        assert len(router.routes) > 0
    
    def test_router_includes_prediction_routes(self):
        """Test that router includes prediction routes"""
        route_paths = [route.path for route in router.routes if hasattr(route, 'path')]
        # Should have prediction-related routes
        assert any('predict' in path.lower() or 'schema' in path.lower() for path in route_paths)

