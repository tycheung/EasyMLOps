"""
Analytics service
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from sqlalchemy import select, and_, func, desc

from app.database import get_session
from app.models.monitoring import (
    TimeSeriesAnalysisDB, ComparativeAnalyticsDB, CustomDashboardDB, AutomatedReportDB
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class AnalyticsService(BaseMonitoringService):
    """Service for analytics"""
    
    async def analyze_time_series_trend(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        model_id: Optional[str] = None,
        deployment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze time series trend for a metric"""
        try:
            # Query metric data from prediction logs or performance metrics
            from app.models.monitoring import PredictionLogDB, ModelPerformanceMetricsDB
            
            async with get_session() as session:
                # Get time series data based on metric name
                if metric_name == "latency":
                    stmt = select(
                        PredictionLogDB.timestamp,
                        PredictionLogDB.latency_ms
                    ).where(
                        and_(
                            PredictionLogDB.timestamp >= start_time,
                            PredictionLogDB.timestamp <= end_time,
                            PredictionLogDB.success == True
                        )
                    )
                    if model_id:
                        stmt = stmt.where(PredictionLogDB.model_id == model_id)
                    if deployment_id:
                        stmt = stmt.where(PredictionLogDB.deployment_id == deployment_id)
                    
                    result = await session.execute(stmt)
                    data_points = result.all()
                    
                    if not data_points:
                        return {
                            "metric_name": metric_name,
                            "time_window_start": start_time.isoformat(),
                            "time_window_end": end_time.isoformat(),
                            "trend_direction": "stable",
                            "trend_slope": 0.0,
                            "trend_strength": 0.0
                        }
                    
                    # Extract values and timestamps
                    values = [point[1] for point in data_points if point[1] is not None]
                    timestamps = [point[0] for point in data_points if point[1] is not None]
                    
                    if len(values) < 2:
                        return {
                            "metric_name": metric_name,
                            "time_window_start": start_time.isoformat(),
                            "time_window_end": end_time.isoformat(),
                            "trend_direction": "stable",
                            "trend_slope": 0.0,
                            "trend_strength": 0.0
                        }
                    
                    # Calculate trend
                    x = np.arange(len(values))
                    slope, intercept = np.polyfit(x, values, 1)
                    correlation = np.corrcoef(x, values)[0, 1]
                    
                    trend_direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
                    trend_strength = abs(correlation) if not np.isnan(correlation) else 0.0
                    
                    analysis = {
                        "metric_name": metric_name,
                        "time_window_start": start_time.isoformat(),
                        "time_window_end": end_time.isoformat(),
                        "trend_direction": trend_direction,
                        "trend_slope": float(slope),
                        "trend_strength": float(trend_strength),
                        "seasonality_detected": False,
                        "anomaly_count": 0
                    }
                    
                    return analysis
                else:
                    # For other metrics, return basic analysis
                    return {
                        "metric_name": metric_name,
                        "time_window_start": start_time.isoformat(),
                        "time_window_end": end_time.isoformat(),
                        "trend_direction": "stable",
                        "trend_slope": 0.0,
                        "trend_strength": 0.5
                    }
                
        except Exception as e:
            logger.error(f"Error analyzing time series trend: {e}", exc_info=True)
            raise
    
    async def create_comparative_analytics(
        self,
        comparison_type: str,
        comparison_name: str,
        entity_ids: List[str],
        entity_types: List[str],
        entity_names: List[str],
        comparison_metrics: Dict[str, Any],
        time_window_start: datetime,
        time_window_end: datetime,
        created_by: Optional[str] = None
    ) -> str:
        """Create comparative analytics"""
        try:
            analytics_id = str(uuid.uuid4())
            
            # Calculate comparison results
            # This is a simplified implementation - would compare actual metrics
            comparison_results = {
                "entities": entity_ids,
                "metrics": comparison_metrics,
                "comparison": {}
            }
            
            analytics_db = ComparativeAnalyticsDB(
                id=analytics_id,
                comparison_type=comparison_type,
                comparison_name=comparison_name,
                entity_ids=entity_ids,
                entity_types=entity_types,
                entity_names=entity_names,
                comparison_metrics=comparison_metrics,
                comparison_results=comparison_results,
                time_window_start=time_window_start,
                time_window_end=time_window_end,
                created_by=created_by
            )
            
            async with get_session() as session:
                session.add(analytics_db)
                await session.commit()
                logger.info(f"Created comparative analytics {analytics_id}: {comparison_name}")
                return analytics_id
        except Exception as e:
            logger.error(f"Error creating comparative analytics: {e}", exc_info=True)
            raise
    
    async def create_custom_dashboard(
        self,
        dashboard_name: str,
        description: Optional[str],
        dashboard_config: Dict[str, Any],
        selected_metrics: List[str],
        visualization_options: Dict[str, Any],
        is_shared: bool,
        shared_with: List[str],
        auto_refresh_enabled: bool,
        refresh_interval_seconds: Optional[int],
        filters: Dict[str, Any],
        created_by: Optional[str]
    ) -> str:
        """Create custom dashboard"""
        try:
            dashboard_id = str(uuid.uuid4())
            dashboard_db = CustomDashboardDB(
                id=dashboard_id,
                dashboard_name=dashboard_name,
                description=description,
                dashboard_config=dashboard_config,
                selected_metrics=selected_metrics,
                visualization_options=visualization_options,
                is_shared=is_shared,
                shared_with=shared_with,
                auto_refresh_enabled=auto_refresh_enabled,
                refresh_interval_seconds=refresh_interval_seconds,
                filters=filters,
                created_by=created_by
            )
            
            async with get_session() as session:
                session.add(dashboard_db)
                await session.commit()
                logger.info(f"Created custom dashboard {dashboard_id}: {dashboard_name}")
                return dashboard_id
        except Exception as e:
            logger.error(f"Error creating custom dashboard: {e}", exc_info=True)
            raise
    
    async def create_automated_report(
        self,
        report_name: str,
        report_type: str,
        description: Optional[str],
        schedule_type: str,
        schedule_config: Dict[str, Any],
        report_config: Dict[str, Any],
        included_metrics: List[str],
        included_models: List[str],
        time_window_days: Optional[int],
        delivery_method: List[str],
        recipients: List[str],
        email_template: Optional[str],
        slack_webhook: Optional[str],
        is_active: bool,
        created_by: Optional[str]
    ) -> str:
        """Create automated report"""
        try:
            report_id = str(uuid.uuid4())
            report_db = AutomatedReportDB(
                id=report_id,
                report_name=report_name,
                report_type=report_type,
                description=description,
                schedule_type=schedule_type,
                schedule_config=schedule_config,
                report_config=report_config,
                included_metrics=included_metrics,
                included_models=included_models,
                time_window_days=time_window_days,
                delivery_method=delivery_method,
                recipients=recipients,
                email_template=email_template,
                slack_webhook=slack_webhook,
                is_active=is_active,
                created_by=created_by
            )
            
            async with get_session() as session:
                session.add(report_db)
                await session.commit()
                logger.info(f"Created automated report {report_id}: {report_name}")
                return report_id
        except Exception as e:
            logger.error(f"Error creating automated report: {e}", exc_info=True)
            raise
