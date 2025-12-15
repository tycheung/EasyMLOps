"""
Tests for app/services/bentoml/builders.py module
Tests the re-export module structure
"""

import pytest
from app.services.bentoml.builders import ServiceBuilder


class TestBentoMLBuildersModule:
    """Test BentoML builders module re-exports"""
    
    def test_service_builder_imported(self):
        """Test ServiceBuilder class is imported"""
        assert ServiceBuilder is not None
        assert isinstance(ServiceBuilder, type)
    
    def test_service_builder_has_methods(self):
        """Test ServiceBuilder has expected methods"""
        # ServiceBuilder is a facade, check it has methods
        methods = [m for m in dir(ServiceBuilder) if not m.startswith('_')]
        assert len(methods) > 0  # Should have some methods
    
    def test_module_exports(self):
        """Test that module has expected exports"""
        import app.services.bentoml.builders as builders_module
        assert hasattr(builders_module, 'ServiceBuilder')

