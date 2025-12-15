"""
Analytics routes
Provides endpoints for time series analysis and comparative analytics
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query, Body
import logging

from app.schemas.monitoring import TimeSeriesAnalysis
from app.services.monitoring_service import monitoring_service
from app.database import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/analytics/time-series", response_model=TimeSeriesAnalysis)
async def analyze_time_series_trend(
    metric_name: str = Query(..., description="Metric name"),
    start_time: datetime = Query(..., description="Analysis period start"),
    end_time: datetime = Query(..., description="Analysis period end"),
    model_id: Optional[str] = Query(None),
    deployment_id: Optional[str] = Query(None)
):
    """Analyze time series trend for a metric"""
    try:
        analysis = await monitoring_service.analyze_time_series_trend(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            deployment_id=deployment_id
        )
        return TimeSeriesAnalysis(**analysis)
    except Exception as e:
        logger.error(f"Error analyzing time series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/comparative", response_model=List[Dict[str, Any]])
async def list_comparative_analytics(
    limit: int = Query(100, description="Maximum number of analytics to return")
):
    """List all comparative analytics"""
    try:
        async with get_session() as session:
            from app.models.monitoring import ComparativeAnalyticsDB
            from sqlalchemy import select, desc
            
            stmt = select(ComparativeAnalyticsDB).order_by(desc(ComparativeAnalyticsDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            analytics_db = result.scalars().all()
            
            return [
                {
                    "id": analytics.id,
                    "comparison_type": analytics.comparison_type,
                    "comparison_name": analytics.comparison_name,
                    "entity_ids": analytics.entity_ids,
                    "entity_types": analytics.entity_types,
                    "entity_names": analytics.entity_names,
                    "comparison_metrics": analytics.comparison_metrics,
                    "time_window_start": analytics.time_window_start.isoformat() if analytics.time_window_start else None,
                    "time_window_end": analytics.time_window_end.isoformat() if analytics.time_window_end else None,
                    "created_at": analytics.created_at.isoformat() if analytics.created_at else None,
                    "created_by": analytics.created_by
                }
                for analytics in analytics_db
            ]
    except Exception as e:
        logger.error(f"Error listing comparative analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/comparative", response_model=Dict[str, str], status_code=201)
async def create_comparative_analytics(analytics: Dict[str, Any]):
    """Create comparative analytics"""
    try:
        from app.schemas.monitoring import ComparativeAnalytics
        analytics_obj = ComparativeAnalytics(**analytics)
        analytics_id = await monitoring_service.create_comparative_analytics(
            comparison_type=analytics_obj.comparison_type,
            comparison_name=analytics_obj.comparison_name,
            entity_ids=analytics_obj.entity_ids,
            entity_types=analytics_obj.entity_types,
            entity_names=analytics_obj.entity_names,
            comparison_metrics=analytics_obj.comparison_metrics,
            time_window_start=analytics_obj.time_window_start,
            time_window_end=analytics_obj.time_window_end,
            created_by=analytics_obj.created_by
        )
        return {"id": analytics_id, "message": "Comparative analytics created successfully"}
    except Exception as e:
        logger.error(f"Error creating comparative analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/dashboards", response_model=List[Dict[str, Any]])
async def list_dashboards(
    is_shared: Optional[bool] = Query(None, description="Filter by shared status"),
    limit: int = Query(100, description="Maximum number of dashboards to return")
):
    """List all custom dashboards"""
    try:
        async with get_session() as session:
            from app.models.monitoring import CustomDashboardDB
            from sqlalchemy import select, desc
            
            stmt = select(CustomDashboardDB)
            if is_shared is not None:
                stmt = stmt.where(CustomDashboardDB.is_shared == is_shared)
            
            stmt = stmt.order_by(desc(CustomDashboardDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            dashboards_db = result.scalars().all()
            
            return [
                {
                    "id": dashboard.id,
                    "dashboard_name": dashboard.dashboard_name,
                    "description": dashboard.description,
                    "dashboard_config": dashboard.dashboard_config,
                    "selected_metrics": dashboard.selected_metrics,
                    "visualization_options": dashboard.visualization_options,
                    "is_shared": dashboard.is_shared,
                    "shared_with": dashboard.shared_with,
                    "auto_refresh_enabled": dashboard.auto_refresh_enabled,
                    "refresh_interval_seconds": dashboard.refresh_interval_seconds,
                    "filters": dashboard.filters,
                    "created_at": dashboard.created_at.isoformat() if dashboard.created_at else None,
                    "created_by": dashboard.created_by
                }
                for dashboard in dashboards_db
            ]
    except Exception as e:
        logger.error(f"Error listing dashboards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/dashboards", response_model=Dict[str, str], status_code=201)
async def create_custom_dashboard(dashboard: Dict[str, Any]):
    """Create custom dashboard"""
    try:
        from app.schemas.monitoring import CustomDashboard
        dashboard_obj = CustomDashboard(**dashboard)
        dashboard_id = await monitoring_service.create_custom_dashboard(
            dashboard_name=dashboard_obj.dashboard_name,
            description=dashboard_obj.description,
            dashboard_config=dashboard_obj.dashboard_config,
            selected_metrics=dashboard_obj.selected_metrics,
            visualization_options=dashboard_obj.visualization_options,
            is_shared=dashboard_obj.is_shared,
            shared_with=dashboard_obj.shared_with,
            auto_refresh_enabled=dashboard_obj.auto_refresh_enabled,
            refresh_interval_seconds=dashboard_obj.refresh_interval_seconds,
            filters=dashboard_obj.filters,
            created_by=dashboard_obj.created_by
        )
        return {"id": dashboard_id, "message": "Custom dashboard created successfully"}
    except Exception as e:
        logger.error(f"Error creating custom dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/reports", response_model=List[Dict[str, Any]])
async def list_reports(
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    report_type: Optional[str] = Query(None, description="Filter by report type"),
    limit: int = Query(100, description="Maximum number of reports to return")
):
    """List all automated reports"""
    try:
        async with get_session() as session:
            from app.models.monitoring import AutomatedReportDB
            from sqlalchemy import select, desc
            
            stmt = select(AutomatedReportDB)
            if is_active is not None:
                stmt = stmt.where(AutomatedReportDB.is_active == is_active)
            if report_type:
                stmt = stmt.where(AutomatedReportDB.report_type == report_type)
            
            stmt = stmt.order_by(desc(AutomatedReportDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            reports_db = result.scalars().all()
            
            return [
                {
                    "id": report.id,
                    "report_name": report.report_name,
                    "report_type": report.report_type,
                    "description": report.description,
                    "schedule_type": report.schedule_type,
                    "schedule_config": report.schedule_config,
                    "report_config": report.report_config,
                    "included_metrics": report.included_metrics,
                    "included_models": report.included_models,
                    "time_window_days": report.time_window_days,
                    "delivery_method": report.delivery_method,
                    "recipients": report.recipients,
                    "is_active": report.is_active,
                    "created_at": report.created_at.isoformat() if report.created_at else None,
                    "created_by": report.created_by
                }
                for report in reports_db
            ]
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/reports", response_model=Dict[str, str], status_code=201)
async def create_automated_report(report: Dict[str, Any]):
    """Create automated report"""
    try:
        from app.schemas.monitoring import AutomatedReport
        report_obj = AutomatedReport(**report)
        report_id = await monitoring_service.create_automated_report(
            report_name=report_obj.report_name,
            report_type=report_obj.report_type,
            description=report_obj.description,
            schedule_type=report_obj.schedule_type,
            schedule_config=report_obj.schedule_config,
            report_config=report_obj.report_config,
            included_metrics=report_obj.included_metrics,
            included_models=report_obj.included_models,
            time_window_days=report_obj.time_window_days,
            delivery_method=report_obj.delivery_method,
            recipients=report_obj.recipients,
            email_template=report_obj.email_template,
            slack_webhook=report_obj.slack_webhook,
            is_active=report_obj.is_active,
            created_by=report_obj.created_by
        )
        return {"id": report_id, "message": "Automated report created successfully"}
    except Exception as e:
        logger.error(f"Error creating automated report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

