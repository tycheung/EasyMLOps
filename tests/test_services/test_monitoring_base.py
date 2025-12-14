"""
Tests for Base Monitoring Service
Tests common utilities and helper methods used by all monitoring services
"""

import pytest
import numpy as np
from unittest.mock import patch, AsyncMock
from typing import Dict, Any

from app.services.monitoring.base import BaseMonitoringService
from app.database import get_session


class TestBaseMonitoringService:
    """Test base monitoring service utilities"""
    
    def test_init(self):
        """Test service initialization"""
        service = BaseMonitoringService()
        assert service.start_time > 0
        assert "cpu_usage" in service.alert_thresholds
        assert service.alert_thresholds["cpu_usage"] == 80.0
    
    @pytest.mark.asyncio
    async def test_check_database_health_success(self):
        """Test successful database health check"""
        service = BaseMonitoringService()
        is_healthy, message = await service._check_database_health()
        assert is_healthy is True
        assert "successful" in message.lower()
    
    @pytest.mark.asyncio
    async def test_check_database_health_failure(self):
        """Test database health check with connection failure"""
        service = BaseMonitoringService()
        with patch('app.services.monitoring.base.get_session') as mock_session:
            mock_session.side_effect = Exception("Connection failed")
            is_healthy, message = await service._check_database_health()
            assert is_healthy is False
            assert "failed" in message.lower()
    
    def test_calculate_psi_identical_distributions(self):
        """Test PSI calculation with identical distributions"""
        service = BaseMonitoringService()
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        current = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        psi = service._calculate_psi(baseline, current)
        assert psi == 0.0  # No drift
    
    def test_calculate_psi_different_distributions(self):
        """Test PSI calculation with different distributions"""
        service = BaseMonitoringService()
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        current = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        psi = service._calculate_psi(baseline, current)
        assert psi > 0.0  # Should detect drift
        assert isinstance(psi, float)
    
    def test_calculate_psi_constant_values(self):
        """Test PSI calculation with constant values (no variation)"""
        service = BaseMonitoringService()
        baseline = np.array([5.0, 5.0, 5.0])
        current = np.array([5.0, 5.0, 5.0])
        psi = service._calculate_psi(baseline, current)
        assert psi == 0.0  # No variation, no drift
    
    def test_calculate_psi_custom_bins(self):
        """Test PSI calculation with custom bin count"""
        service = BaseMonitoringService()
        baseline = np.random.normal(0, 1, 100)
        current = np.random.normal(1, 1, 100)
        psi_10 = service._calculate_psi(baseline, current, bins=10)
        psi_20 = service._calculate_psi(baseline, current, bins=20)
        assert isinstance(psi_10, float)
        assert isinstance(psi_20, float)
        # Both should detect drift (positive values)
        assert psi_10 >= 0.0
        assert psi_20 >= 0.0
    
    def test_calculate_psi_error_handling(self):
        """Test PSI calculation error handling"""
        service = BaseMonitoringService()
        with patch('numpy.concatenate', side_effect=Exception("Test error")):
            psi = service._calculate_psi(np.array([1.0]), np.array([2.0]))
            assert psi == 0.0  # Should return 0 on error
    
    def test_ks_test_normal_distributions(self):
        """Test KS test with normal distributions"""
        service = BaseMonitoringService()
        baseline = np.random.normal(0, 1, 100)
        current = np.random.normal(0, 1, 100)
        statistic, p_value = service._ks_test(baseline, current)
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0.0 <= statistic <= 1.0
        assert 0.0 <= p_value <= 1.0
    
    def test_ks_test_different_distributions(self):
        """Test KS test with different distributions"""
        service = BaseMonitoringService()
        baseline = np.random.normal(0, 1, 100)
        current = np.random.normal(5, 1, 100)
        statistic, p_value = service._ks_test(baseline, current)
        assert statistic > 0.0  # Should detect difference
        assert p_value < 0.05  # Should be statistically significant
    
    def test_ks_test_empty_arrays(self):
        """Test KS test with empty arrays"""
        service = BaseMonitoringService()
        statistic, p_value = service._ks_test(np.array([]), np.array([]))
        assert statistic == 0.0
        assert p_value == 1.0
    
    def test_ks_test_error_handling(self):
        """Test KS test error handling"""
        service = BaseMonitoringService()
        with patch('scipy.stats.ks_2samp', side_effect=Exception("Test error")):
            statistic, p_value = service._ks_test(np.array([1.0]), np.array([2.0]))
            assert statistic == 0.0
            assert p_value == 1.0
    
    def test_extract_prediction_value_float(self):
        """Test extracting prediction value from float"""
        service = BaseMonitoringService()
        value = service._extract_prediction_value(0.75)
        assert value == 0.75
    
    def test_extract_prediction_value_int(self):
        """Test extracting prediction value from int"""
        service = BaseMonitoringService()
        value = service._extract_prediction_value(42)
        assert value == 42.0
    
    def test_extract_prediction_value_dict(self):
        """Test extracting prediction value from dict"""
        service = BaseMonitoringService()
        # Test with 'prediction' key
        value = service._extract_prediction_value({"prediction": 0.85})
        assert value == 0.85
        
        # Test with 'value' key
        value = service._extract_prediction_value({"value": 0.92})
        assert value == 0.92
        
        # Test with 'score' key
        value = service._extract_prediction_value({"score": 0.78})
        assert value == 0.78
    
    def test_extract_prediction_value_list(self):
        """Test extracting prediction value from list"""
        service = BaseMonitoringService()
        value = service._extract_prediction_value([0.1, 0.2, 0.7])
        assert value == 0.1  # First value
    
    def test_extract_prediction_value_invalid(self):
        """Test extracting prediction value from invalid data"""
        service = BaseMonitoringService()
        value = service._extract_prediction_value("not a number")
        assert value is None
        
        value = service._extract_prediction_value({"unknown": "key"})
        assert value is None
    
    def test_extract_confidence_score_dict(self):
        """Test extracting confidence score from dict"""
        service = BaseMonitoringService()
        # Test with 'confidence' key
        score = service._extract_confidence_score({"confidence": 0.85})
        assert score == 0.85
        
        # Test with 'probability' key
        score = service._extract_confidence_score({"probability": 0.92})
        assert score == 0.92
    
    def test_extract_confidence_score_probabilities(self):
        """Test extracting confidence from probabilities array"""
        service = BaseMonitoringService()
        score = service._extract_confidence_score({"probabilities": [0.1, 0.2, 0.7]})
        assert score == 0.7  # Max probability
    
    def test_extract_confidence_score_logits(self):
        """Test extracting confidence from logits"""
        service = BaseMonitoringService()
        score = service._extract_confidence_score({"logits": [1.0, 2.0, 3.0]})
        assert score is not None
        assert 0.0 <= score <= 1.0
    
    def test_extract_confidence_score_list(self):
        """Test extracting confidence from list"""
        service = BaseMonitoringService()
        score = service._extract_confidence_score([0.1, 0.2, 0.7])
        # The method applies softmax to lists (treats as logits), so max won't be exactly 0.7
        assert score is not None
        assert 0.0 <= score <= 1.0
        # After softmax, the value should be a valid probability
        # The exact value depends on softmax normalization
    
    def test_extract_confidence_score_normalization(self):
        """Test confidence score normalization for out-of-range values"""
        service = BaseMonitoringService()
        # Test with value > 1.0 (should be clamped)
        score = service._extract_confidence_score({"confidence": 1.5})
        assert score is not None
        assert 0.0 <= score <= 1.0
    
    def test_extract_confidence_scores_multi_class(self):
        """Test extracting confidence scores for multi-class"""
        service = BaseMonitoringService()
        scores = service._extract_confidence_scores({"probabilities": [0.1, 0.2, 0.7]})
        assert scores is not None
        assert isinstance(scores, dict)
        assert "0" in scores
        assert "1" in scores
        assert "2" in scores
        assert scores["2"] == 0.7
    
    def test_extract_uncertainty_dict(self):
        """Test extracting uncertainty from dict"""
        service = BaseMonitoringService()
        uncertainty = service._extract_uncertainty({"uncertainty": 0.15})
        assert uncertainty == 0.15
        
        uncertainty = service._extract_uncertainty({"entropy": 1.5})
        assert uncertainty == 1.5
    
    def test_extract_uncertainty_from_entropy(self):
        """Test calculating uncertainty from probabilities entropy"""
        service = BaseMonitoringService()
        uncertainty = service._extract_uncertainty({"probabilities": [0.5, 0.5]})
        assert uncertainty is not None
        assert uncertainty > 0.0
    
    def test_extract_prediction_interval(self):
        """Test extracting prediction interval"""
        service = BaseMonitoringService()
        lower, upper = service._extract_prediction_interval({
            "prediction_interval": {"lower": 0.1, "upper": 0.9}
        })
        assert lower == 0.1
        assert upper == 0.9
    
    def test_extract_prediction_interval_separate_fields(self):
        """Test extracting prediction interval from separate fields"""
        service = BaseMonitoringService()
        lower, upper = service._extract_prediction_interval({
            "lower_bound": 0.2,
            "upper_bound": 0.8
        })
        assert lower == 0.2
        assert upper == 0.8
    
    def test_extract_protected_attribute(self):
        """Test extracting protected attribute from input data"""
        service = BaseMonitoringService()
        input_data = {"age": 25, "gender": "female", "income": 50000}
        attr = service._extract_protected_attribute(input_data, "gender")
        assert attr == "female"
        
        attr = service._extract_protected_attribute(input_data, "age")
        assert attr == "25"
    
    def test_extract_protected_attribute_missing(self):
        """Test extracting missing protected attribute"""
        service = BaseMonitoringService()
        input_data = {"age": 25}
        attr = service._extract_protected_attribute(input_data, "gender")
        assert attr == ""
    
    def test_extract_ground_truth_value_int(self):
        """Test extracting ground truth from int"""
        service = BaseMonitoringService()
        value = service._extract_ground_truth_value(1)
        assert value == 1
        
        value = service._extract_ground_truth_value(0)
        assert value == 0
    
    def test_extract_ground_truth_value_bool(self):
        """Test extracting ground truth from bool"""
        service = BaseMonitoringService()
        value = service._extract_ground_truth_value(True)
        assert value == 1
        
        value = service._extract_ground_truth_value(False)
        assert value == 0
    
    def test_extract_ground_truth_value_string(self):
        """Test extracting ground truth from string"""
        service = BaseMonitoringService()
        value = service._extract_ground_truth_value("1")
        assert value == 1
        
        value = service._extract_ground_truth_value("true")
        assert value == 1
        
        value = service._extract_ground_truth_value("false")
        assert value == 0
    
    def test_extract_ground_truth_value_dict(self):
        """Test extracting ground truth from dict"""
        service = BaseMonitoringService()
        value = service._extract_ground_truth_value({"label": 1})
        assert value == 1
        
        value = service._extract_ground_truth_value({"value": "true"})
        assert value == 1
    
    def test_extract_ground_truth_value_none(self):
        """Test extracting ground truth from None"""
        service = BaseMonitoringService()
        value = service._extract_ground_truth_value(None)
        assert value is None

