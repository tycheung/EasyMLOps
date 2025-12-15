"""
Tests for monitoring routes module
Tests re-exports and module structure
"""

import pytest
from app.routes.monitoring import router
from app.routes.monitoring.middleware import log_prediction_middleware


class TestMonitoringModule:
    """Test monitoring routes module exports"""
    
    def test_router_exported(self):
        """Test that router is exported"""
        assert router is not None
        assert hasattr(router, 'routes')
    
    def test_log_prediction_middleware_exported(self):
        """Test that log_prediction_middleware function is exported"""
        assert log_prediction_middleware is not None
        assert callable(log_prediction_middleware)
    
    def test_router_has_routes(self):
        """Test that router has registered routes"""
        # Router should have monitoring routes
        assert len(router.routes) > 0
    
    def test_router_includes_monitoring_routes(self):
        """Test that router includes monitoring-related routes"""
        route_paths = [route.path for route in router.routes if hasattr(route, 'path')]
        # Should have monitoring-related routes
        assert len(route_paths) > 0


class TestLogPredictionMiddleware:
    """Test log_prediction_middleware function"""
    
    @pytest.mark.asyncio
    async def test_log_prediction_middleware_success(self):
        """Test middleware logs prediction successfully"""
        from unittest.mock import MagicMock, AsyncMock, patch
        from fastapi import Request
        
        mock_request = MagicMock(spec=Request)
        mock_request.url = "http://localhost:8000/predict"
        mock_request.headers = {"user-agent": "test-agent"}
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        
        with patch('app.services.monitoring_service.monitoring_service.log_prediction') as mock_log:
            mock_log.return_value = AsyncMock()
            
            await log_prediction_middleware(
                request=mock_request,
                model_id="test_model_123",
                input_data={"feature1": 0.5},
                output_data={"prediction": 0.8},
                latency_ms=45.2,
                success=True
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]['model_id'] == "test_model_123"
            assert call_args[1]['success'] is True
            assert call_args[1]['latency_ms'] == 45.2
    
    @pytest.mark.asyncio
    async def test_log_prediction_middleware_with_error(self):
        """Test middleware logs prediction with error"""
        from unittest.mock import MagicMock, AsyncMock, patch
        from fastapi import Request
        
        mock_request = MagicMock(spec=Request)
        mock_request.url = "http://localhost:8000/predict"
        mock_request.headers = {"user-agent": "test-agent"}
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        
        with patch('app.services.monitoring_service.monitoring_service.log_prediction') as mock_log:
            mock_log.return_value = AsyncMock()
            
            await log_prediction_middleware(
                request=mock_request,
                model_id="test_model_123",
                input_data={"feature1": 0.5},
                output_data=None,
                latency_ms=45.2,
                success=False,
                error_message="Test error"
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]['success'] is False
            assert call_args[1]['error_message'] == "Test error"
    
    @pytest.mark.asyncio
    async def test_log_prediction_middleware_handles_exception(self):
        """Test middleware handles exceptions gracefully"""
        from unittest.mock import MagicMock, patch
        from fastapi import Request
        
        mock_request = MagicMock(spec=Request)
        mock_request.url = "http://localhost:8000/predict"
        mock_request.headers = {"user-agent": "test-agent"}
        mock_request.client = None  # No client
        
        with patch('app.services.monitoring_service.monitoring_service.log_prediction') as mock_log:
            mock_log.side_effect = Exception("Service error")
            
            # Should not raise exception
            await log_prediction_middleware(
                request=mock_request,
                model_id="test_model_123",
                input_data={"feature1": 0.5},
                output_data={"prediction": 0.8},
                latency_ms=45.2,
                success=True
            )
            
            # Should have attempted to log
            mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_prediction_middleware_no_client(self):
        """Test middleware handles missing client"""
        from unittest.mock import MagicMock, AsyncMock, patch
        from fastapi import Request
        
        mock_request = MagicMock(spec=Request)
        mock_request.url = "http://localhost:8000/predict"
        mock_request.headers = {"user-agent": "test-agent"}
        mock_request.client = None
        
        with patch('app.services.monitoring_service.monitoring_service.log_prediction') as mock_log:
            mock_log.return_value = AsyncMock()
            
            await log_prediction_middleware(
                request=mock_request,
                model_id="test_model_123",
                input_data={"feature1": 0.5},
                output_data={"prediction": 0.8},
                latency_ms=45.2
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]['ip_address'] is None

