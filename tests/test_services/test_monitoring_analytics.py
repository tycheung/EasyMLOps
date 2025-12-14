"""
Tests for Analytics Service
Tests time-series analysis, comparative analytics, dashboards, and automated reports
"""

import pytest
from datetime import datetime, timedelta
import uuid

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import (
    AnalysisType, ComparisonType, ReportType, ScheduleType
)
from app.database import get_session


class TestAnalyticsService:
    """Test analytics service functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_time_series_trend(self, test_model):
        """Test time-series trend analysis"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=48)
        end_time = now
        
        # Create prediction logs with increasing latency trend
        async with get_session() as session:
            for i in range(48):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": float(i)},
                    output_data={"prediction": i % 2},
                    latency_ms=50.0 + i * 0.5,  # Increasing trend
                    timestamp=start_time + timedelta(hours=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Analyze trend
        analysis = await monitoring_service.analyze_time_series(
            metric_name="latency",
            start_time=start_time,
            end_time=end_time,
            analysis_type=AnalysisType.TREND,
            model_id=test_model.id
        )
        
        assert analysis is not None
        assert analysis.analysis_type == AnalysisType.TREND
        assert analysis.metric_name == "latency"
        assert analysis.trend_direction is not None
        assert analysis.trend_slope is not None
        assert analysis.trend_strength is not None
        assert 0 <= analysis.trend_strength <= 1
    
    @pytest.mark.asyncio
    async def test_create_comparative_analytics(self, test_model):
        """Test creating comparative analytics"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=24)
        end_time = now
        
        # Create prediction logs for the model
        async with get_session() as session:
            for i in range(100):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": float(i)},
                    output_data={"prediction": i % 2},
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i * 100),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Create comparative analytics
        analytics = await monitoring_service.create_comparative_analytics(
            comparison_type=ComparisonType.MODEL_COMPARISON,
            comparison_name="Model Performance Comparison",
            entity_ids=[test_model.id],
            entity_types=["model"],
            start_time=start_time,
            end_time=end_time,
            comparison_metrics={"primary_metric": "avg_latency_ms"}
        )
        
        assert analytics is not None
        assert analytics.comparison_type == ComparisonType.MODEL_COMPARISON
        assert analytics.comparison_name == "Model Performance Comparison"
        assert len(analytics.entity_ids) == 1
        assert analytics.entity_ids[0] == test_model.id
    
    @pytest.mark.asyncio
    async def test_create_custom_dashboard(self, test_model):
        """Test creating a custom dashboard"""
        dashboard = await monitoring_service.create_custom_dashboard(
            dashboard_name="Test Dashboard",
            selected_metrics=["latency", "error_rate", "throughput"],
            dashboard_config={"layout": "grid", "widgets": []},
            description="Test dashboard description",
            created_by="test_user"
        )
        
        assert dashboard is not None
        assert dashboard.dashboard_name == "Test Dashboard"
        assert len(dashboard.selected_metrics) == 3
        assert "latency" in dashboard.selected_metrics
        assert dashboard.created_by == "test_user"
        assert dashboard.is_shared is False
    
    @pytest.mark.asyncio
    async def test_create_automated_report(self, test_model):
        """Test creating an automated report"""
        report = await monitoring_service.create_automated_report(
            report_name="Daily Performance Report",
            report_type=ReportType.DAILY,
            schedule_type=ScheduleType.DAILY,
            report_config={"include_metrics": ["latency", "error_rate"]},
            delivery_method=["email"],
            recipients=["admin@example.com"],
            description="Daily performance metrics report",
            created_by="admin"
        )
        
        assert report is not None
        assert report.report_name == "Daily Performance Report"
        assert report.report_type == ReportType.DAILY
        assert report.schedule_type == ScheduleType.DAILY
        assert report.is_active is True
        assert len(report.recipients) == 1
        assert "email" in report.delivery_method

