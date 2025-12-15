"""
Comprehensive tests for drift detection routes
Tests all drift detection endpoints with various scenarios
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from app.schemas.monitoring import ModelDriftDetection, DriftType, DriftSeverity


class TestDriftDetectionRoutes:
    """Test drift detection route endpoints"""
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.detect_feature_drift', new_callable=AsyncMock)
    @patch('app.services.monitoring_service.monitoring_service.store_drift_detection', new_callable=AsyncMock)
    async def test_detect_feature_drift_success(self, mock_store, mock_detect, client):
        """Test successful feature drift detection"""
        mock_result = ModelDriftDetection(
            model_id="test_model",
            drift_type=DriftType.FEATURE,
            detection_method="psi",
            baseline_window_start=datetime.utcnow() - timedelta(days=7),
            baseline_window_end=datetime.utcnow() - timedelta(days=1),
            current_window_start=datetime.utcnow() - timedelta(days=1),
            current_window_end=datetime.utcnow(),
            drift_detected=True,
            drift_score=0.75,
            drift_severity=DriftSeverity.HIGH
        )
        mock_detect.return_value = mock_result
        mock_store.return_value = "drift_id_123"
        
        baseline_start = (datetime.utcnow() - timedelta(days=7)).isoformat()
        baseline_end = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_start = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_end = datetime.utcnow().isoformat()
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/drift/feature",
            params={
                "baseline_window_start": baseline_start,
                "baseline_window_end": baseline_end,
                "current_window_start": current_start,
                "current_window_end": current_end,
                "drift_threshold": 0.2,
                "ks_p_value_threshold": 0.05
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["drift_detected"] is True
        assert result["drift_type"] == DriftType.FEATURE.value
        mock_detect.assert_called_once()
        mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.detect_data_drift', new_callable=AsyncMock)
    @patch('app.services.monitoring_service.monitoring_service.store_drift_detection', new_callable=AsyncMock)
    async def test_detect_data_drift_success(self, mock_store, mock_detect, client):
        """Test successful data drift detection"""
        mock_result = ModelDriftDetection(
            model_id="test_model",
            drift_type=DriftType.DATA,
            detection_method="schema_comparison",
            baseline_window_start=datetime.utcnow() - timedelta(days=7),
            baseline_window_end=datetime.utcnow() - timedelta(days=1),
            current_window_start=datetime.utcnow() - timedelta(days=1),
            current_window_end=datetime.utcnow(),
            drift_detected=True,
            drift_score=0.6,
            drift_severity=DriftSeverity.MEDIUM
        )
        mock_detect.return_value = mock_result
        mock_store.return_value = "drift_id_456"
        
        baseline_start = (datetime.utcnow() - timedelta(days=7)).isoformat()
        baseline_end = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_start = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_end = datetime.utcnow().isoformat()
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/drift/data",
            params={
                "baseline_window_start": baseline_start,
                "baseline_window_end": baseline_end,
                "current_window_start": current_start,
                "current_window_end": current_end,
                "deployment_id": "deploy_123"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["drift_detected"] is True
        assert result["drift_type"] == DriftType.DATA.value
        mock_detect.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.detect_prediction_drift', new_callable=AsyncMock)
    @patch('app.services.monitoring_service.monitoring_service.store_drift_detection', new_callable=AsyncMock)
    async def test_detect_prediction_drift_success(self, mock_store, mock_detect, client):
        """Test successful prediction drift detection"""
        mock_result = ModelDriftDetection(
            model_id="test_model",
            drift_type=DriftType.PREDICTION,
            detection_method="ks_test",
            baseline_window_start=datetime.utcnow() - timedelta(days=7),
            baseline_window_end=datetime.utcnow() - timedelta(days=1),
            current_window_start=datetime.utcnow() - timedelta(days=1),
            current_window_end=datetime.utcnow(),
            drift_detected=True,
            drift_score=0.8,
            drift_severity=DriftSeverity.HIGH,
            prediction_mean_shift=0.15,
            prediction_variance_shift=0.2
        )
        mock_detect.return_value = mock_result
        mock_store.return_value = "drift_id_789"
        
        baseline_start = (datetime.utcnow() - timedelta(days=7)).isoformat()
        baseline_end = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_start = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_end = datetime.utcnow().isoformat()
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/drift/prediction",
            params={
                "baseline_window_start": baseline_start,
                "baseline_window_end": baseline_end,
                "current_window_start": current_start,
                "current_window_end": current_end,
                "drift_threshold": 0.2
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["drift_detected"] is True
        assert result["drift_type"] == DriftType.PREDICTION.value
        assert result["prediction_mean_shift"] == 0.15
        mock_detect.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.detect_feature_drift', new_callable=AsyncMock)
    async def test_detect_feature_drift_error(self, mock_detect, client):
        """Test feature drift detection with error"""
        mock_detect.side_effect = Exception("Drift detection failed")
        
        baseline_start = (datetime.utcnow() - timedelta(days=7)).isoformat()
        baseline_end = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_start = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_end = datetime.utcnow().isoformat()
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/drift/feature",
            params={
                "baseline_window_start": baseline_start,
                "baseline_window_end": baseline_end,
                "current_window_start": current_start,
                "current_window_end": current_end
            }
        )
        
        assert response.status_code == 500
        result = response.json()
        # Custom exception handler uses {"error": {"message": ...}} format
        assert "error" in result
        assert "Drift detection failed" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.database.get_session')
    async def test_get_drift_history_success(self, mock_get_session, client, test_model, test_session):
        """Test getting drift history"""
        from app.models.monitoring import ModelDriftDetectionDB
        
        drift_record = ModelDriftDetectionDB(
            id="drift_123",
            model_id=test_model.id,
            drift_type="feature",
            detection_method="psi",
            baseline_window_start=datetime.utcnow() - timedelta(days=7),
            baseline_window_end=datetime.utcnow() - timedelta(days=1),
            current_window_start=datetime.utcnow() - timedelta(days=1),
            current_window_end=datetime.utcnow(),
            drift_detected=True,
            drift_score=0.75,
            drift_severity="high",
            timestamp=datetime.utcnow()
        )
        test_session.add(drift_record)
        test_session.commit()
        
        response = client.get(
            f"/api/v1/monitoring/models/{test_model.id}/drift",
            params={"limit": 10}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        if len(result) > 0:
            assert result[0]["model_id"] == test_model.id
    
    @pytest.mark.asyncio
    @patch('app.database.get_session')
    async def test_get_drift_history_with_filters(self, mock_get_session, client, test_model, test_session):
        """Test getting drift history with filters"""
        from app.models.monitoring import ModelDriftDetectionDB
        
        drift_record = ModelDriftDetectionDB(
            id="drift_456",
            model_id=test_model.id,
            deployment_id="deploy_123",
            drift_type="prediction",
            detection_method="ks_test",
            baseline_window_start=datetime.utcnow() - timedelta(days=7),
            baseline_window_end=datetime.utcnow() - timedelta(days=1),
            current_window_start=datetime.utcnow() - timedelta(days=1),
            current_window_end=datetime.utcnow(),
            drift_detected=True,
            drift_score=0.6,
            drift_severity="medium",
            timestamp=datetime.utcnow()
        )
        test_session.add(drift_record)
        test_session.commit()
        
        response = client.get(
            f"/api/v1/monitoring/models/{test_model.id}/drift",
            params={
                "deployment_id": "deploy_123",
                "drift_type": "prediction",
                "limit": 10
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    @patch('app.database.get_session')
    async def test_get_drift_history_empty(self, mock_get_session, client, test_session):
        """Test getting drift history when none exists"""
        response = client.get(
            "/api/v1/monitoring/models/nonexistent_model/drift",
            params={"limit": 10}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        assert len(result) == 0
    
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.detect_data_drift', new_callable=AsyncMock)
    async def test_detect_data_drift_error(self, mock_detect, client):
        """Test data drift detection with error"""
        mock_detect.side_effect = Exception("Database error")
        
        baseline_start = (datetime.utcnow() - timedelta(days=7)).isoformat()
        baseline_end = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_start = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_end = datetime.utcnow().isoformat()
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/drift/data",
            params={
                "baseline_window_start": baseline_start,
                "baseline_window_end": baseline_end,
                "current_window_start": current_start,
                "current_window_end": current_end
            }
        )
        
        assert response.status_code == 500
        result = response.json()
        assert "Database error" in str(result)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.detect_prediction_drift', new_callable=AsyncMock)
    async def test_detect_prediction_drift_error(self, mock_detect, client):
        """Test prediction drift detection with error"""
        mock_detect.side_effect = Exception("Calculation error")
        
        baseline_start = (datetime.utcnow() - timedelta(days=7)).isoformat()
        baseline_end = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_start = (datetime.utcnow() - timedelta(days=1)).isoformat()
        current_end = datetime.utcnow().isoformat()
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/drift/prediction",
            params={
                "baseline_window_start": baseline_start,
                "baseline_window_end": baseline_end,
                "current_window_start": current_start,
                "current_window_end": current_end
            }
        )
        
        assert response.status_code == 500
        result = response.json()
        assert "Calculation error" in str(result)
    
    @pytest.mark.asyncio
    @patch('app.routes.monitoring.drift.get_session')
    async def test_get_drift_history_error(self, mock_get_session, client):
        """Test getting drift history with error"""
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(side_effect=Exception("Database error"))
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        response = client.get("/api/v1/monitoring/models/test_model/drift")
        
        assert response.status_code == 500
        result = response.json()
        assert "Database error" in str(result)

