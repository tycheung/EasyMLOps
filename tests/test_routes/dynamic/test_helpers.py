"""
Tests for dynamic route helper functions
Tests prediction helpers, simulation helpers, and logging helpers
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from app.models.model import ModelDeployment
from app.routes.dynamic.prediction_helpers import (
    make_prediction_call,
    make_batch_prediction_call,
    make_proba_prediction_call
)
from app.routes.dynamic.simulation_helpers import (
    simulate_sklearn_prediction,
    simulate_tensorflow_prediction,
    simulate_pytorch_prediction,
    simulate_boosting_prediction,
    simulate_generic_prediction
)
from app.routes.dynamic.logging_helpers import log_prediction


class TestSimulationHelpers:
    """Test prediction simulation helpers"""
    
    def test_simulate_sklearn_prediction_list_of_dicts(self):
        """Test sklearn simulation with list of dicts"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        request_data = {
            "data": [
                {"feature1": 1.0, "feature2": 2.0},
                {"feature1": 3.0, "feature2": 4.0}
            ]
        }
        
        result = simulate_sklearn_prediction(deployment, request_data)
        
        assert "predictions" in result
        assert len(result["predictions"]) == 2
        assert result["model_id"] == "test_model"
        assert result["deployment_id"] == "test_deployment"
        assert result["framework"] == "sklearn"
    
    def test_simulate_sklearn_prediction_single_dict(self):
        """Test sklearn simulation with single dict"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        request_data = {"data": {"feature1": 1.0, "feature2": 2.0}}
        
        result = simulate_sklearn_prediction(deployment, request_data)
        
        assert "predictions" in result
        assert len(result["predictions"]) == 1
        assert result["predictions"][0] == 0.85
    
    def test_simulate_tensorflow_prediction(self):
        """Test TensorFlow simulation"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="tensorflow"
        )
        request_data = {"data": [[1.0, 2.0, 3.0]]}
        
        result = simulate_tensorflow_prediction(deployment, request_data)
        
        assert "predictions" in result
        assert isinstance(result["predictions"][0], list)
        assert result["framework"] == "tensorflow"
    
    def test_simulate_pytorch_prediction(self):
        """Test PyTorch simulation"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="pytorch"
        )
        request_data = {"data": [1.0, 2.0, 3.0]}
        
        result = simulate_pytorch_prediction(deployment, request_data)
        
        assert "predictions" in result
        assert result["predictions"][0] == 0.85
        assert result["framework"] == "pytorch"
    
    def test_simulate_boosting_prediction(self):
        """Test boosting (XGBoost/LightGBM) simulation"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="xgboost"
        )
        request_data = {"data": [1.0, 2.0]}
        
        result = simulate_boosting_prediction(deployment, request_data)
        
        assert "predictions" in result
        assert result["predictions"][0] == 0.92
        assert result["framework"] == "xgboost"
    
    def test_simulate_generic_prediction(self):
        """Test generic prediction simulation"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="unknown"
        )
        request_data = {"data": "anything"}
        
        result = simulate_generic_prediction(deployment, request_data)
        
        assert "predictions" in result
        assert result["predictions"][0] == 0.5
        assert result["framework"] == "unknown"


class TestPredictionHelpers:
    """Test prediction helper functions"""
    
    @pytest.mark.asyncio
    async def test_make_prediction_call_sklearn(self):
        """Test making prediction call for sklearn"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        request_data = {"data": [{"feature1": 1.0}]}
        
        result = await make_prediction_call(deployment, request_data)
        
        assert "predictions" in result
        assert result["model_id"] == "test_model"
        assert result["framework"] == "sklearn"
    
    @pytest.mark.asyncio
    async def test_make_prediction_call_tensorflow(self):
        """Test making prediction call for TensorFlow"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="tensorflow"
        )
        request_data = {"data": [[1.0, 2.0]]}
        
        result = await make_prediction_call(deployment, request_data)
        
        assert "predictions" in result
        assert result["framework"] == "tensorflow"
    
    @pytest.mark.asyncio
    async def test_make_prediction_call_pytorch(self):
        """Test making prediction call for PyTorch"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="pytorch"
        )
        request_data = {"data": [1.0, 2.0]}
        
        result = await make_prediction_call(deployment, request_data)
        
        assert "predictions" in result
        assert result["framework"] == "pytorch"
    
    @pytest.mark.asyncio
    async def test_make_prediction_call_xgboost(self):
        """Test making prediction call for XGBoost"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="xgboost"
        )
        request_data = {"data": [1.0, 2.0]}
        
        result = await make_prediction_call(deployment, request_data)
        
        assert "predictions" in result
        assert result["framework"] == "xgboost"
    
    @pytest.mark.asyncio
    async def test_make_prediction_call_lightgbm(self):
        """Test making prediction call for LightGBM"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="lightgbm"
        )
        request_data = {"data": [1.0, 2.0]}
        
        result = await make_prediction_call(deployment, request_data)
        
        assert "predictions" in result
        assert result["framework"] == "lightgbm"
    
    @pytest.mark.asyncio
    async def test_make_prediction_call_generic(self):
        """Test making prediction call for unknown framework"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="unknown"
        )
        request_data = {"data": "anything"}
        
        result = await make_prediction_call(deployment, request_data)
        
        assert "predictions" in result
        assert result["framework"] == "unknown"
    
    @pytest.mark.asyncio
    async def test_make_prediction_call_error_handling(self):
        """Test error handling in prediction call"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        request_data = {"data": [{"feature1": 1.0}]}
        
        with patch('app.routes.dynamic.prediction_helpers.simulate_sklearn_prediction') as mock_sim:
            mock_sim.side_effect = ValueError("Test error")
            
            with pytest.raises(ValueError):
                await make_prediction_call(deployment, request_data)
    
    @pytest.mark.asyncio
    async def test_make_batch_prediction_call(self):
        """Test making batch prediction call"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        request_data = {
            "data": [
                {"feature1": 1.0},
                {"feature1": 2.0},
                {"feature1": 3.0}
            ]
        }
        
        result = await make_batch_prediction_call(deployment, request_data)
        
        assert "predictions" in result
        assert len(result["predictions"]) == 3
        assert result["batch_size"] == 3
        assert result["model_id"] == "test_model"
    
    @pytest.mark.asyncio
    async def test_make_batch_prediction_call_error(self):
        """Test error handling in batch prediction call"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        request_data = {}  # Missing "data" key
        
        with pytest.raises(KeyError):
            await make_batch_prediction_call(deployment, request_data)
    
    @pytest.mark.asyncio
    async def test_make_proba_prediction_call(self):
        """Test making probability prediction call"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        request_data = {"data": [{"feature1": 1.0}]}
        
        result = await make_proba_prediction_call(deployment, request_data)
        
        assert "probabilities" in result
        assert isinstance(result["probabilities"], list)
        assert result["model_id"] == "test_model"
    
    @pytest.mark.asyncio
    async def test_make_proba_prediction_call_error(self):
        """Test error handling in probability prediction call"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        request_data = {"data": [{"feature1": 1.0}]}
        
        with patch('app.routes.dynamic.prediction_helpers.logger') as mock_logger:
            # Should not raise, but should log error
            result = await make_proba_prediction_call(deployment, request_data)
            assert "probabilities" in result


class TestLoggingHelpers:
    """Test logging helper functions"""
    
    @pytest.mark.asyncio
    async def test_log_prediction_success(self):
        """Test successful prediction logging"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        
        session = AsyncMock()
        session.get = AsyncMock(return_value=deployment)
        
        request_data = {"data": [{"feature1": 1.0}]}
        response_data = {"predictions": [0.75]}
        
        await log_prediction(
            session,
            "test_deployment",
            request_data,
            response_data,
            is_batch=False,
            endpoint="predict"
        )
        
        session.get.assert_called_once()
        session.add.assert_called_once()
        session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_prediction_deployment_not_found(self):
        """Test logging when deployment not found"""
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)
        
        request_data = {"data": [{"feature1": 1.0}]}
        response_data = {"predictions": [0.75]}
        
        with patch('app.routes.dynamic.logging_helpers.logger') as mock_logger:
            await log_prediction(
                session,
                "nonexistent_deployment",
                request_data,
                response_data
            )
            
            mock_logger.warning.assert_called_once()
            session.add.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_log_prediction_error_handling(self):
        """Test error handling in prediction logging"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        
        session = AsyncMock()
        session.get = AsyncMock(return_value=deployment)
        session.add.side_effect = Exception("Database error")
        session.commit.side_effect = Exception("Database error")
        
        request_data = {"data": [{"feature1": 1.0}]}
        response_data = {"predictions": [0.75]}
        
        with patch('app.routes.dynamic.logging_helpers.logger') as mock_logger:
            # Should not raise exception
            await log_prediction(
                session,
                "test_deployment",
                request_data,
                response_data
            )
            
            # Error should be logged when commit fails
            mock_logger.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_log_prediction_batch(self):
        """Test logging batch predictions"""
        deployment = ModelDeployment(
            id="test_deployment",
            model_id="test_model",
            framework="sklearn"
        )
        
        session = AsyncMock()
        session.get = AsyncMock(return_value=deployment)
        
        request_data = {"data": [{"feature1": 1.0}, {"feature1": 2.0}]}
        response_data = {"predictions": [0.75, 0.85]}
        
        await log_prediction(
            session,
            "test_deployment",
            request_data,
            response_data,
            is_batch=True,
            endpoint="predict_batch"
        )
        
        session.add.assert_called_once()
        added_log = session.add.call_args[0][0]
        assert added_log.is_batch is True
        assert added_log.api_endpoint == "predict_batch"

