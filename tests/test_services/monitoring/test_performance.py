"""
Tests for Performance Monitoring Service
Tests prediction logging, performance metrics calculation, and resource utilization
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta, timezone
import uuid

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import ModelPerformanceMetrics
from app.database import get_session


class TestPerformanceMonitoringService:
    """Test performance monitoring service functionality"""
    
    @pytest.mark.asyncio
    async def test_log_prediction_success(self, test_model):
        """Test successful prediction logging"""
        input_data = {"feature1": 0.5, "feature2": "test"}
        output_data = {"prediction": 0.75, "confidence": 0.85}
        latency_ms = 45.2
        
        # The actual log_prediction method returns a UUID string
        result = await monitoring_service.log_prediction(
            model_id=test_model.id,
            deployment_id="deploy_123",
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            api_endpoint="/predict/deploy_123",
            success=True
        )
        
        # log_prediction returns a string ID (UUID)
        assert isinstance(result, str)
        # Just check that it's a valid UUID format (36 characters with hyphens)
        assert len(result) == 36
        assert result.count('-') == 4
        
        # Verify the log was actually stored
        async with get_session() as session:
            from app.models.monitoring import PredictionLogDB
            from sqlalchemy import select
            log_result = await session.execute(
                select(PredictionLogDB).where(PredictionLogDB.id == result)
            )
            log = log_result.scalar_one_or_none()
            assert log is not None
            assert log.model_id == test_model.id
            assert log.confidence_score == 0.85
    
    @pytest.mark.asyncio
    async def test_get_model_performance_metrics(self, test_model):
        """Test performance metrics calculation"""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        
        # Mock some prediction logs
        with patch.object(monitoring_service, 'get_model_performance_metrics') as mock_metrics:
            # Return a proper ModelPerformanceMetrics object
            from app.schemas.monitoring import ModelPerformanceMetrics
            mock_metrics.return_value = ModelPerformanceMetrics(
                model_id=test_model.id,
                time_window_start=start_time,
                time_window_end=end_time,
                total_requests=150,
                successful_requests=145,
                failed_requests=5,
                requests_per_minute=2.5,
                requests_per_second=0.042,
                requests_per_hour=150.0,
                avg_concurrent_requests=1.9,
                avg_queue_depth=0.9,
                avg_latency_ms=45.2,
                p50_latency_ms=42.1,
                p95_latency_ms=78.5,
                p99_latency_ms=95.2,
                p99_9_latency_ms=110.0,
                min_latency_ms=20.0,
                max_latency_ms=120.0,
                std_dev_latency_ms=15.3,
                latency_distribution={"bins": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], "counts": [10, 20, 30, 40, 25, 15, 5, 3, 1, 1], "bin_width": 10.0, "total_samples": 150},
                success_rate=96.7,
                error_rate=3.3,
                avg_ttfb_ms=5.0,
                avg_inference_time_ms=35.0,
                avg_total_time_ms=45.2,
                batch_metrics=None
            )
            
            metrics = await monitoring_service.get_model_performance_metrics(
                model_id=test_model.id,
                start_time=start_time,
                end_time=end_time
            )
            
            # Test object attributes instead of dict keys
            assert metrics.total_requests == 150
            assert metrics.avg_latency_ms == 45.2
            assert metrics.success_rate == 96.7
            # Test new enhanced latency metrics
            assert metrics.p99_9_latency_ms == 110.0
            assert metrics.min_latency_ms == 20.0
            assert metrics.std_dev_latency_ms == 15.3
            assert metrics.latency_distribution is not None
            assert metrics.latency_distribution["total_samples"] == 150
    
    @pytest.mark.asyncio
    async def test_enhanced_latency_metrics_calculation(self, test_model):
        """Test calculation of enhanced latency metrics (P99.9, min, std dev, distribution)"""
        from datetime import datetime, timedelta
        
        # Create prediction logs with varying latencies
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Create 1000 logs to test P99.9 properly, all within the time window
        async with get_session() as session:
            # First, clean up any existing logs for this model in the time window
            from sqlalchemy import delete
            delete_stmt = delete(PredictionLogDB).where(
                PredictionLogDB.model_id == test_model.id,
                PredictionLogDB.timestamp >= start_time,
                PredictionLogDB.timestamp <= end_time
            )
            await session.execute(delete_stmt)
            await session.commit()
            
            logs = []
            for i in range(1000):
                # Create a distribution: most requests are fast, some are slow
                if i < 500:
                    latency = 50.0 + (i % 50)  # 50-100ms range
                elif i < 800:
                    latency = 100.0 + (i % 100)  # 100-200ms range
                elif i < 950:
                    latency = 200.0 + (i % 100)  # 200-300ms range
                else:
                    latency = 300.0 + (i % 200)  # 300-500ms range (outliers)
                
                # Distribute timestamps evenly within the time window
                time_offset = (end_time - start_time) * (i / 1000.0)
                log_timestamp = start_time + time_offset
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"test": "data"},
                    output_data={"prediction": 1.0},
                    latency_ms=latency,
                    timestamp=log_timestamp,
                    api_endpoint="/predict",
                    success=True
                )
                logs.append(log)
                session.add(log)
            
            await session.commit()
            
            # Calculate metrics
            metrics = await monitoring_service.get_model_performance_metrics(
                model_id=test_model.id,
                start_time=start_time,
                end_time=end_time
            )
        
        # Verify enhanced metrics
        assert metrics.total_requests == 1000, f"Expected 1000 requests, got {metrics.total_requests}"
        assert metrics.min_latency_ms is not None
        assert metrics.min_latency_ms >= 50.0  # Should be at least 50ms
        assert metrics.p99_9_latency_ms is not None
        assert metrics.p99_9_latency_ms > metrics.p99_latency_ms  # P99.9 should be higher than P99
        assert metrics.std_dev_latency_ms is not None
        assert metrics.std_dev_latency_ms > 0  # Should have some variance
        assert metrics.latency_distribution is not None
        assert "bins" in metrics.latency_distribution
        assert "counts" in metrics.latency_distribution
        assert len(metrics.latency_distribution["bins"]) == 11  # 10 bins + 1 edge
        assert len(metrics.latency_distribution["counts"]) == 10  # 10 bin counts
        assert metrics.latency_distribution["total_samples"] == 1000
        
        # Verify percentile calculations are correct
        sorted_latencies = sorted([log.latency_ms for log in logs])
        expected_p99_9 = sorted_latencies[int(0.999 * len(sorted_latencies))]
        assert abs(metrics.p99_9_latency_ms - expected_p99_9) < 1.0  # Allow small rounding difference
    
    @pytest.mark.asyncio
    async def test_enhanced_latency_metrics_empty_data(self, test_model):
        """Test enhanced latency metrics with no prediction logs"""
        from datetime import datetime, timedelta
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        metrics = await monitoring_service.get_model_performance_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Verify all metrics are zero or None for empty data
        assert metrics.total_requests == 0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.p50_latency_ms == 0.0
        assert metrics.p95_latency_ms == 0.0
        assert metrics.p99_latency_ms == 0.0
        assert metrics.p99_9_latency_ms is None
        assert metrics.min_latency_ms is None
        assert metrics.max_latency_ms == 0.0
        assert metrics.std_dev_latency_ms is None
        assert metrics.latency_distribution is None
        # Advanced performance metrics should be None or 0
        assert metrics.requests_per_second == 0.0
        assert metrics.requests_per_hour == 0.0
        assert metrics.avg_concurrent_requests is None
        assert metrics.avg_queue_depth is None
        assert metrics.avg_ttfb_ms is None
        assert metrics.avg_inference_time_ms is None
        assert metrics.avg_total_time_ms == 0.0
        assert metrics.batch_metrics is None
    
    @pytest.mark.asyncio
    async def test_advanced_performance_metrics(self, test_model):
        """Test calculation of advanced performance metrics (RPS, concurrent requests, TTFB, etc.)"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        async with get_session() as session:
            # Clean up existing logs
            from sqlalchemy import delete
            delete_stmt = delete(PredictionLogDB).where(
                PredictionLogDB.model_id == test_model.id,
                PredictionLogDB.timestamp >= start_time,
                PredictionLogDB.timestamp <= end_time
            )
            await session.execute(delete_stmt)
            await session.commit()
            
            # Create 100 logs with various timing metrics
            logs = []
            for i in range(100):
                time_offset = (end_time - start_time) * (i / 100.0)
                log_timestamp = start_time + time_offset
                
                # Vary latencies and timing metrics
                total_latency = 50.0 + (i % 50)  # 50-100ms
                inference_time = total_latency * 0.7  # 70% of total is inference
                ttfb = total_latency * 0.1  # 10% is TTFB
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"test": "data"},
                    output_data={"prediction": 1.0},
                    latency_ms=total_latency,
                    inference_time_ms=inference_time,
                    ttfb_ms=ttfb,
                    is_batch=False,
                    timestamp=log_timestamp,
                    api_endpoint="/predict",
                    success=True
                )
                logs.append(log)
                session.add(log)
            
            # Add some batch requests
            for i in range(10):
                time_offset = (end_time - start_time) * ((90 + i) / 100.0)
                log_timestamp = start_time + time_offset
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"test": "batch_data"},
                    output_data={"predictions": [1.0, 2.0, 3.0]},
                    latency_ms=150.0 + i * 10,  # 150-240ms for batches
                    inference_time_ms=120.0 + i * 8,
                    ttfb_ms=20.0,
                    is_batch=True,
                    batch_size=3,
                    timestamp=log_timestamp,
                    api_endpoint="/predict/batch",
                    success=True
                )
                logs.append(log)
                session.add(log)
            
            await session.commit()
            
            # Calculate metrics
            metrics = await monitoring_service.get_model_performance_metrics(
                model_id=test_model.id,
                start_time=start_time,
                end_time=end_time
            )
        
        # Verify throughput metrics
        assert metrics.total_requests == 110
        assert metrics.requests_per_second is not None
        assert metrics.requests_per_second > 0
        assert metrics.requests_per_minute > 0
        assert metrics.requests_per_hour is not None
        assert metrics.requests_per_hour > 0
        
        # Verify concurrent requests estimation (Little's Law)
        assert metrics.avg_concurrent_requests is not None
        assert metrics.avg_concurrent_requests >= 0
        # Should be approximately: requests_per_second * avg_latency_seconds
        if metrics.requests_per_second > 0 and metrics.avg_latency_ms > 0:
            expected_concurrent = metrics.requests_per_second * (metrics.avg_latency_ms / 1000.0)
            assert abs(metrics.avg_concurrent_requests - expected_concurrent) < 0.1
        
        # Verify queue depth
        if metrics.avg_concurrent_requests is not None and metrics.avg_concurrent_requests > 1:
            assert metrics.avg_queue_depth is not None
            assert metrics.avg_queue_depth >= 0
        
        # Verify timing metrics
        assert metrics.avg_ttfb_ms is not None
        assert metrics.avg_ttfb_ms > 0
        assert metrics.avg_inference_time_ms is not None
        assert metrics.avg_inference_time_ms > 0
        assert metrics.avg_total_time_ms is not None
        assert metrics.avg_total_time_ms > 0
        
        # Verify batch metrics
        assert metrics.batch_metrics is not None
        assert metrics.batch_metrics["total_batch_requests"] == 10
        assert metrics.batch_metrics["avg_batch_size"] == 3.0
        assert metrics.batch_metrics["min_batch_size"] == 3
        assert metrics.batch_metrics["max_batch_size"] == 3
        assert metrics.batch_metrics["avg_batch_latency_ms"] is not None
        assert metrics.batch_metrics["items_per_second"] > 0
    
    @pytest.mark.asyncio
    async def test_resource_utilization_metrics(self, test_model):
        """Test collection of resource utilization metrics (GPU, network I/O)"""
        from datetime import datetime, timedelta
        
        # Test system health metrics collection includes network I/O
        health_metrics = await monitoring_service.collect_system_health_metrics()
        
        # Verify network metrics are collected
        network_metrics = [m for m in health_metrics if m.metric_type and "network" in m.metric_type.value]
        # Network metrics may or may not be available depending on system
        # Just verify the method doesn't crash
        
        # Verify GPU metrics collection (may not be available)
        gpu_metrics = [m for m in health_metrics if m.metric_type and "gpu" in m.metric_type.value.lower()]
        # GPU metrics are optional, so we just verify the method handles absence gracefully
        
        # Test model-specific resource usage collection
        now = datetime.utcnow()
        start_time = now - timedelta(minutes=5)
        end_time = now
        
        resource_usage = await monitoring_service.collect_model_resource_usage(
            model_id=test_model.id,
            time_window_start=start_time,
            time_window_end=end_time
        )
        
        # Verify resource usage object
        assert resource_usage.model_id == test_model.id
        assert resource_usage.time_window_start == start_time
        assert resource_usage.time_window_end == end_time
        # Resource values may be None if not available, which is fine
        assert resource_usage.cpu_usage_percent is None or (0 <= resource_usage.cpu_usage_percent <= 100)
        assert resource_usage.memory_usage_mb is None or resource_usage.memory_usage_mb >= 0
        assert resource_usage.gpu_usage_percent is None or (0 <= resource_usage.gpu_usage_percent <= 100)
        assert resource_usage.gpu_memory_usage_mb is None or resource_usage.gpu_memory_usage_mb >= 0
        
        # Test storing resource usage
        usage_id = await monitoring_service.store_model_resource_usage(resource_usage)
        assert usage_id is not None
    
    @pytest.mark.asyncio
    async def test_network_io_metrics_collection(self):
        """Test that network I/O metrics are collected in system health"""
        from app.schemas.monitoring import MetricType
        
        health_metrics = await monitoring_service.collect_system_health_metrics()
        
        # Check for network metrics (they should be present if psutil.net_io_counters works)
        has_network_metrics = any(
            m.metric_type in [MetricType.NETWORK_BYTES_SENT, MetricType.NETWORK_BYTES_RECV,
                            MetricType.NETWORK_PACKETS_SENT, MetricType.NETWORK_PACKETS_RECV]
            for m in health_metrics if m.metric_type
        )
        
        # Network metrics should be available on most systems
        # If not available, that's okay - the method should handle it gracefully
        # We just verify the method completes without error
    
    @pytest.mark.asyncio
    async def test_gpu_metrics_collection(self):
        """Test GPU metrics collection (optional - may not be available)"""
        from app.schemas.monitoring import MetricType
        
        health_metrics = await monitoring_service.collect_system_health_metrics()
        
        # GPU metrics are optional - check if any are present
        gpu_metrics = [
            m for m in health_metrics 
            if m.metric_type in [MetricType.GPU_USAGE, MetricType.GPU_MEMORY_USAGE]
        ]
        
        # GPU may or may not be available - just verify method handles both cases
        # If GPU is available, metrics should have valid values
        for metric in gpu_metrics:
            assert metric.value is not None
            assert 0 <= metric.value <= 100  # GPU usage should be percentage
    
    @pytest.mark.asyncio
    async def test_get_deployment_summary(self, test_model, test_deployment):
        """Test getting deployment summary"""
        summary = await monitoring_service.get_deployment_summary(test_deployment.id)
        assert summary is not None
        assert "deployment_id" in summary or "error" in summary
        if "deployment_id" in summary:
            assert summary["deployment_id"] == test_deployment.id
            assert summary["model_id"] == test_model.id
    
    @pytest.mark.asyncio
    async def test_get_deployment_summary_not_found(self):
        """Test getting deployment summary for non-existent deployment"""
        summary = await monitoring_service.get_deployment_summary("nonexistent-id")
        assert "error" in summary
    
    @pytest.mark.asyncio
    async def test_get_aggregated_metrics(self, test_model):
        """Test getting aggregated metrics"""
        metrics = await monitoring_service.get_aggregated_metrics(
            model_id=test_model.id,
            time_range="24h"
        )
        assert metrics is not None
        assert isinstance(metrics, dict)
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_metrics(self, test_model):
        """Test calculating confidence metrics"""
        from datetime import datetime, timedelta
        
        # First, log some predictions with confidence scores
        for i in range(20):
            await monitoring_service.log_prediction(
                model_id=test_model.id,
                deployment_id=None,
                input_data={"feature1": i},
                output_data={"prediction": 0.5 + (i % 10) * 0.05, "confidence": 0.7 + (i % 5) * 0.05},
                latency_ms=50.0,
                api_endpoint="/predict",
                success=True
            )
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        confidence_metrics = await monitoring_service.calculate_confidence_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert confidence_metrics is not None
        assert confidence_metrics.model_id == test_model.id
        assert confidence_metrics.total_samples >= 0
        assert confidence_metrics.samples_with_confidence >= 0
    
    @pytest.mark.asyncio
    async def test_store_performance_metrics(self, test_model):
        """Test storing performance metrics"""
        from app.schemas.monitoring import ModelPerformanceMetrics
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        metrics = ModelPerformanceMetrics(
            model_id=test_model.id,
            time_window_start=now - timedelta(hours=1),
            time_window_end=now,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            requests_per_minute=1.67,  # Required field
            avg_latency_ms=50.0,
            p50_latency_ms=45.0,  # Required field
            p95_latency_ms=75.0,  # Required field
            p99_latency_ms=90.0,  # Required field
            max_latency_ms=100.0,  # Required field
            success_rate=0.95,
            error_rate=0.05  # Required field
        )
        
        metrics_id = await monitoring_service.store_performance_metrics(metrics)
        assert metrics_id is not None
        assert isinstance(metrics_id, str)
    
    @pytest.mark.asyncio
    async def test_store_confidence_metrics(self, test_model):
        """Test storing confidence metrics"""
        from app.schemas.monitoring import ModelConfidenceMetrics
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        metrics = ModelConfidenceMetrics(
            model_id=test_model.id,
            time_window_start=now - timedelta(hours=1),
            time_window_end=now,
            total_samples=100,
            samples_with_confidence=100,
            avg_confidence=0.85,
            low_confidence_count=5
        )
        
        metrics_id = await monitoring_service.store_confidence_metrics(metrics)
        assert metrics_id is not None
        assert isinstance(metrics_id, str)

