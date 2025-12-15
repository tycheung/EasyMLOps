"""
Tests for app/routes/monitoring.py module
Tests the re-export module structure
"""

import pytest
from app.routes.monitoring import router


class TestMonitoringRoutesModule:
    """Test monitoring routes module re-exports"""
    
    def test_router_imported(self):
        """Test router is imported from monitoring package"""
        assert router is not None
        assert hasattr(router, 'routes')
    
    def test_module_exports(self):
        """Test that module has expected exports"""
        import app.routes.monitoring as monitoring_module
        assert hasattr(monitoring_module, 'router')
    
    def test_router_has_monitoring_routes(self):
        """Test router includes monitoring routes"""
        assert len(router.routes) > 0

