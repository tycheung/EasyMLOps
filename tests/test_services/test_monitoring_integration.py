"""
Tests for Integration Service
Tests external integrations, webhooks, sampling, and metric aggregation configurations
"""

import pytest
from datetime import datetime

from app.services.monitoring_service import monitoring_service
from app.schemas.monitoring import (
    IntegrationType, SamplingStrategy, AggregationMethod
)


class TestIntegrationService:
    """Test integration service functionality"""
    
    @pytest.mark.asyncio
    async def test_create_external_integration(self, test_model):
        """Test creating an external integration"""
        integration = await monitoring_service.create_external_integration(
            integration_type=IntegrationType.PROMETHEUS,
            integration_name="Prometheus Export",
            config={"export_path": "/metrics", "port": 9090},
            endpoint_url="http://prometheus:9090",
            description="Prometheus metrics export",
            created_by="admin"
        )
        
        assert integration is not None
        assert integration.integration_type == IntegrationType.PROMETHEUS
        assert integration.integration_name == "Prometheus Export"
        assert integration.config["export_path"] == "/metrics"
        assert integration.is_active is True
    
    @pytest.mark.asyncio
    async def test_create_webhook_config(self, test_model):
        """Test creating a webhook configuration"""
        webhook = await monitoring_service.create_webhook_config(
            webhook_name="Alert Webhook",
            webhook_url="https://example.com/webhook",
            trigger_events=["alert_created", "drift_detected"],
            description="Webhook for alerts",
            created_by="admin"
        )
        
        assert webhook is not None
        assert webhook.webhook_name == "Alert Webhook"
        assert webhook.webhook_url == "https://example.com/webhook"
        assert len(webhook.trigger_events) == 2
        assert "alert_created" in webhook.trigger_events
        assert webhook.is_active is True
    
    @pytest.mark.asyncio
    async def test_create_sampling_config(self, test_model):
        """Test creating a sampling configuration"""
        config = await monitoring_service.create_sampling_config(
            config_name="High Volume Sampling",
            resource_type="prediction_log",
            sampling_strategy=SamplingStrategy.RANDOM,
            sampling_rate=0.1,  # Sample 10%
            min_volume_threshold=1000,
            model_id=test_model.id,
            created_by="admin"
        )
        
        assert config is not None
        assert config.config_name == "High Volume Sampling"
        assert config.resource_type == "prediction_log"
        assert config.sampling_strategy == SamplingStrategy.RANDOM
        assert config.sampling_rate == 0.1
        assert config.min_volume_threshold == 1000
        assert config.model_id == test_model.id
        assert config.is_active is True
    
    @pytest.mark.asyncio
    async def test_create_metric_aggregation_config(self, test_model):
        """Test creating a metric aggregation configuration"""
        config = await monitoring_service.create_metric_aggregation_config(
            config_name="Latency Aggregation",
            metric_type="latency",
            aggregation_window_seconds=300,  # 5 minutes
            aggregation_method=AggregationMethod.AVG,
            raw_data_retention_hours=24,
            aggregated_data_retention_days=90,
            model_id=test_model.id,
            created_by="admin"
        )
        
        assert config is not None
        assert config.config_name == "Latency Aggregation"
        assert config.metric_type == "latency"
        assert config.aggregation_window_seconds == 300
        assert config.aggregation_method == AggregationMethod.AVG
        assert config.raw_data_retention_hours == 24
        assert config.aggregated_data_retention_days == 90
        assert config.model_id == test_model.id
        assert config.is_active is True
    
    @pytest.mark.asyncio
    async def test_should_sample(self, test_model):
        """Test sampling decision logic"""
        # Create a sampling config
        await monitoring_service.create_sampling_config(
            config_name="Test Sampling",
            resource_type="prediction_log",
            sampling_strategy=SamplingStrategy.RANDOM,
            sampling_rate=0.5,  # 50% sampling
            model_id=test_model.id
        )
        
        # Test sampling decision
        should_sample, rate = await monitoring_service.should_sample(
            resource_type="prediction_log",
            model_id=test_model.id,
            current_volume=500
        )
        
        assert isinstance(should_sample, bool)
        assert rate == 0.5

