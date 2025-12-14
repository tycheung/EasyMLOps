"""
AB testing service
Handles A/B test creation, variant assignment, metrics calculation, and statistical analysis
"""

import logging
import random
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats
from sqlalchemy import select

from app.database import get_session
from app.models.monitoring import ABTestAssignmentDB, ABTestDB, ABTestMetricsDB, PredictionLogDB
from app.schemas.monitoring import ABTest, ABTestComparison, ABTestMetrics, ABTestStatus
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class ABTestingService(BaseMonitoringService):
    """Service for AB testing"""
    
    async def create_ab_test(
        self,
        test_name: str,
        model_name: str,
        variant_a_model_id: str,
        variant_b_model_id: str,
        variant_a_percentage: float = 50.0,
        variant_b_percentage: float = 50.0,
        use_sticky_sessions: bool = False,
        variant_a_deployment_id: Optional[str] = None,
        variant_b_deployment_id: Optional[str] = None,
        min_sample_size: Optional[int] = None,
        significance_level: float = 0.05,
        primary_metric: str = "accuracy",
        description: Optional[str] = None,
        scheduled_start: Optional[datetime] = None,
        scheduled_end: Optional[datetime] = None,
        created_by: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> ABTest:
        """Create a new A/B test configuration"""
        try:
            # Validate percentages sum to 100
            if abs(variant_a_percentage + variant_b_percentage - 100.0) > 0.01:
                raise ValueError("Variant percentages must sum to 100")
            
            ab_test = ABTest(
                test_name=test_name,
                description=description,
                model_name=model_name,
                variant_a_model_id=variant_a_model_id,
                variant_b_model_id=variant_b_model_id,
                variant_a_deployment_id=variant_a_deployment_id,
                variant_b_deployment_id=variant_b_deployment_id,
                variant_a_percentage=variant_a_percentage,
                variant_b_percentage=variant_b_percentage,
                use_sticky_sessions=use_sticky_sessions,
                min_sample_size=min_sample_size,
                significance_level=significance_level,
                primary_metric=primary_metric,
                scheduled_start=scheduled_start,
                scheduled_end=scheduled_end,
                created_by=created_by,
                config=config or {}
            )
            
            # Store test and get the stored ID
            stored_id = await self.store_ab_test(ab_test)
            ab_test.id = stored_id
            
            logger.info(f"Created A/B test: {test_name}")
            return ab_test
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}", exc_info=True)
            raise
    
    async def store_ab_test(self, ab_test: ABTest) -> str:
        """Store A/B test in database"""
        try:
            ab_test_db = ABTestDB(
                id=str(uuid.uuid4()),
                test_name=ab_test.test_name,
                description=ab_test.description,
                model_name=ab_test.model_name,
                variant_a_model_id=ab_test.variant_a_model_id,
                variant_b_model_id=ab_test.variant_b_model_id,
                variant_a_deployment_id=ab_test.variant_a_deployment_id,
                variant_b_deployment_id=ab_test.variant_b_deployment_id,
                variant_a_percentage=ab_test.variant_a_percentage,
                variant_b_percentage=ab_test.variant_b_percentage,
                use_sticky_sessions=ab_test.use_sticky_sessions,
                status=ab_test.status.value if hasattr(ab_test.status, 'value') else ab_test.status,
                scheduled_start=ab_test.scheduled_start,
                scheduled_end=ab_test.scheduled_end,
                min_sample_size=ab_test.min_sample_size,
                significance_level=ab_test.significance_level,
                primary_metric=ab_test.primary_metric,
                variant_a_samples=ab_test.variant_a_samples if hasattr(ab_test, 'variant_a_samples') else 0,
                variant_b_samples=ab_test.variant_b_samples if hasattr(ab_test, 'variant_b_samples') else 0,
                winner=ab_test.winner if hasattr(ab_test, 'winner') else None,
                conclusion_reason=ab_test.conclusion_reason if hasattr(ab_test, 'conclusion_reason') else None,
                p_value=ab_test.p_value if hasattr(ab_test, 'p_value') else None,
                confidence_interval_lower=ab_test.confidence_interval_lower if hasattr(ab_test, 'confidence_interval_lower') else None,
                confidence_interval_upper=ab_test.confidence_interval_upper if hasattr(ab_test, 'confidence_interval_upper') else None,
                is_statistically_significant=ab_test.is_statistically_significant if hasattr(ab_test, 'is_statistically_significant') else None,
                config=ab_test.config,
                created_by=ab_test.created_by
            )
            
            async with get_session() as session:
                session.add(ab_test_db)
                await session.commit()
                logger.info(f"Stored A/B test {ab_test_db.id}")
                return ab_test_db.id
                
        except Exception as e:
            logger.error(f"Error storing A/B test: {e}")
            raise
    
    async def assign_variant(
        self,
        test_id: str,
        session_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Assign a variant to a session/user for A/B testing"""
        try:
            async with get_session() as session:
                # Check for existing assignment (sticky sessions)
                stmt = select(ABTestAssignmentDB).where(
                    ABTestAssignmentDB.test_id == test_id,
                    ABTestAssignmentDB.session_id == session_id
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()
                
                if existing:
                    return existing.assigned_variant
                
                # Get test configuration
                test = await session.get(ABTestDB, test_id)
                if not test:
                    raise ValueError(f"A/B test {test_id} not found")
                
                if test.status != "running":
                    raise ValueError(f"A/B test {test_id} is not running")
                
                # Assign variant based on percentage
                rand = random.random() * 100.0
                
                if rand < test.variant_a_percentage:
                    assigned_variant = "variant_a"
                else:
                    assigned_variant = "variant_b"
                
                # Store assignment
                assignment = ABTestAssignmentDB(
                    id=str(uuid.uuid4()),
                    test_id=test_id,
                    session_id=session_id,
                    user_id=user_id,
                    assigned_variant=assigned_variant
                )
                
                session.add(assignment)
                await session.commit()
                
                logger.debug(f"Assigned {assigned_variant} to session {session_id} for test {test_id}")
                return assigned_variant
                
        except Exception as e:
            logger.error(f"Error assigning variant: {e}")
            raise
    
    async def start_ab_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        try:
            async with get_session() as session:
                test = await session.get(ABTestDB, test_id)
                if not test:
                    raise ValueError(f"A/B test {test_id} not found")
                
                if test.status not in ["draft", "paused"]:
                    raise ValueError(f"Cannot start test in status: {test.status}")
                
                test.status = "running"
                test.start_time = datetime.utcnow()
                test.updated_at = datetime.utcnow()
                
                await session.commit()
                logger.info(f"Started A/B test {test_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error starting A/B test: {e}")
            raise
    
    async def stop_ab_test(self, test_id: str) -> bool:
        """Stop an A/B test"""
        try:
            async with get_session() as session:
                test = await session.get(ABTestDB, test_id)
                if not test:
                    raise ValueError(f"A/B test {test_id} not found")
                
                test.status = "completed"
                test.end_time = datetime.utcnow()
                test.updated_at = datetime.utcnow()
                
                # Calculate final results
                await self._calculate_ab_test_results(test_id)
                
                await session.commit()
                logger.info(f"Stopped A/B test {test_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping A/B test: {e}")
            raise
    
    async def calculate_ab_test_metrics(
        self,
        test_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> ABTestComparison:
        """Calculate and compare metrics for both variants in an A/B test"""
        try:
            async with get_session() as session:
                test = await session.get(ABTestDB, test_id)
                if not test:
                    raise ValueError(f"A/B test {test_id} not found")
                
                # Get metrics for variant A
                variant_a_metrics = await self._get_variant_metrics(
                    test_id=test_id,
                    variant="variant_a",
                    model_id=test.variant_a_model_id,
                    deployment_id=test.variant_a_deployment_id,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Get metrics for variant B
                variant_b_metrics = await self._get_variant_metrics(
                    test_id=test_id,
                    variant="variant_b",
                    model_id=test.variant_b_model_id,
                    deployment_id=test.variant_b_deployment_id,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Calculate deltas
                comparison = ABTestComparison(
                    test_id=test_id,
                    variant_a_metrics=variant_a_metrics,
                    variant_b_metrics=variant_b_metrics,
                    accuracy_delta=variant_b_metrics.accuracy - variant_a_metrics.accuracy if variant_b_metrics.accuracy and variant_a_metrics.accuracy else None,
                    precision_delta=variant_b_metrics.precision - variant_a_metrics.precision if variant_b_metrics.precision and variant_a_metrics.precision else None,
                    recall_delta=variant_b_metrics.recall - variant_a_metrics.recall if variant_b_metrics.recall and variant_a_metrics.recall else None,
                    f1_delta=variant_b_metrics.f1_score - variant_a_metrics.f1_score if variant_b_metrics.f1_score and variant_a_metrics.f1_score else None,
                    mae_delta=variant_b_metrics.mae - variant_a_metrics.mae if variant_b_metrics.mae and variant_a_metrics.mae else None,
                    r2_delta=variant_b_metrics.r2_score - variant_a_metrics.r2_score if variant_b_metrics.r2_score and variant_a_metrics.r2_score else None,
                    latency_delta_ms=variant_b_metrics.avg_latency_ms - variant_a_metrics.avg_latency_ms if variant_b_metrics.avg_latency_ms and variant_a_metrics.avg_latency_ms else None
                )
                
                # Perform statistical significance testing
                await self._calculate_statistical_significance(comparison, test.primary_metric, test.significance_level)
                
                # Determine winner
                comparison.winner = self._determine_winner(comparison, test.primary_metric)
                comparison.recommendation = self._generate_recommendation(comparison, test.primary_metric)
                
                return comparison
                
        except Exception as e:
            logger.error(f"Error calculating A/B test metrics: {e}", exc_info=True)
            raise
    
    async def _get_variant_metrics(
        self,
        test_id: str,
        variant: str,
        model_id: str,
        deployment_id: Optional[str],
        start_time: datetime,
        end_time: datetime
    ) -> ABTestMetrics:
        """Get aggregated metrics for a variant"""
        try:
            # Note: get_model_performance_metrics, calculate_classification_metrics,
            # calculate_regression_metrics, and calculate_confidence_metrics will be
            # accessed via service composition
            # For now, we'll create metrics with placeholder values
            
            # Get prediction logs for this variant
            async with get_session() as session:
                stmt = select(PredictionLogDB).where(
                    PredictionLogDB.model_id == model_id,
                    PredictionLogDB.timestamp >= start_time,
                    PredictionLogDB.timestamp <= end_time,
                    PredictionLogDB.success == True
                )
                if deployment_id:
                    stmt = stmt.where(PredictionLogDB.deployment_id == deployment_id)
                
                result = await session.execute(stmt)
                logs = result.scalars().all()
            
            total_requests = len(logs)
            successful_requests = sum(1 for log in logs if log.success)
            failed_requests = total_requests - successful_requests
            
            metrics = ABTestMetrics(
                test_id=test_id,
                variant=variant,
                model_id=model_id,
                deployment_id=deployment_id,
                time_window_start=start_time,
                time_window_end=end_time,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                avg_latency_ms=None,  # Will be set via service composition
                p50_latency_ms=None,
                p95_latency_ms=None,
                p99_latency_ms=None,
                error_rate=(failed_requests / total_requests * 100.0) if total_requests > 0 else 0.0
            )
            
            # Store metrics
            await self.store_ab_test_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting variant metrics: {e}")
            raise
    
    async def _calculate_statistical_significance(
        self,
        comparison: ABTestComparison,
        primary_metric: str,
        significance_level: float
    ):
        """Calculate statistical significance using appropriate test"""
        try:
            # Get the primary metric values
            if primary_metric == "accuracy":
                metric_a = comparison.variant_a_metrics.accuracy
                metric_b = comparison.variant_b_metrics.accuracy
                n_a = comparison.variant_a_metrics.total_requests
                n_b = comparison.variant_b_metrics.total_requests
            elif primary_metric == "f1_score":
                metric_a = comparison.variant_a_metrics.f1_score
                metric_b = comparison.variant_b_metrics.f1_score
                n_a = comparison.variant_a_metrics.total_requests
                n_b = comparison.variant_b_metrics.total_requests
            elif primary_metric == "mae":
                metric_a = comparison.variant_a_metrics.mae
                metric_b = comparison.variant_b_metrics.mae
                n_a = comparison.variant_a_metrics.total_requests
                n_b = comparison.variant_b_metrics.total_requests
            else:
                metric_a = comparison.variant_a_metrics.accuracy
                metric_b = comparison.variant_b_metrics.accuracy
                n_a = comparison.variant_a_metrics.total_requests
                n_b = comparison.variant_b_metrics.total_requests
            
            if metric_a is None or metric_b is None or n_a < 10 or n_b < 10:
                comparison.is_statistically_significant = False
                return
            
            # Perform two-proportion z-test for classification metrics
            try:
                if primary_metric in ["accuracy", "f1_score", "precision", "recall"]:
                    if primary_metric == "accuracy":
                        successes_a = int(metric_a * n_a)
                        successes_b = int(metric_b * n_b)
                        
                        p_pool = (successes_a + successes_b) / (n_a + n_b)
                        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
                        
                        if se > 0:
                            z_score = (metric_b - metric_a) / se
                            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                        else:
                            p_value = 1.0
                    else:
                        se = np.sqrt((metric_a * (1 - metric_a) / n_a) + (metric_b * (1 - metric_b) / n_b))
                        if se > 0:
                            t_score = (metric_b - metric_a) / se
                            p_value = 2 * (1 - stats.norm.cdf(abs(t_score)))
                        else:
                            p_value = 1.0
                else:
                    # For regression metrics, use t-test
                    se = np.sqrt((1/n_a) + (1/n_b))
                    if se > 0:
                        t_score = (metric_b - metric_a) / se
                        df = n_a + n_b - 2
                        p_value = 2 * (1 - stats.t.cdf(abs(t_score), df))
                    else:
                        p_value = 1.0
                
                comparison.p_value = float(p_value)
                comparison.is_statistically_significant = p_value < significance_level
                
                # Calculate confidence interval (95%)
                diff = metric_b - metric_a
                if se > 0:
                    z_critical = stats.norm.ppf(0.975)
                    margin = z_critical * se
                    comparison.confidence_interval_lower = float(diff - margin)
                    comparison.confidence_interval_upper = float(diff + margin)
                else:
                    comparison.confidence_interval_lower = diff
                    comparison.confidence_interval_upper = diff
                    
            except Exception as e:
                logger.debug(f"Error calculating statistical significance: {e}")
                comparison.is_statistically_significant = False
                
        except Exception as e:
            logger.error(f"Error in statistical significance calculation: {e}")
            comparison.is_statistically_significant = False
    
    def _determine_winner(
        self,
        comparison: ABTestComparison,
        primary_metric: str
    ) -> str:
        """Determine the winner of the A/B test"""
        try:
            if not comparison.is_statistically_significant:
                return "inconclusive"
            
            if primary_metric in ["accuracy", "f1_score", "precision", "recall", "auc_roc", "r2_score"]:
                # Higher is better
                if comparison.accuracy_delta is not None and primary_metric == "accuracy":
                    if comparison.accuracy_delta > 0:
                        return "variant_b"
                    elif comparison.accuracy_delta < 0:
                        return "variant_a"
                elif comparison.f1_delta is not None and primary_metric == "f1_score":
                    if comparison.f1_delta > 0:
                        return "variant_b"
                    elif comparison.f1_delta < 0:
                        return "variant_a"
                elif comparison.r2_delta is not None and primary_metric == "r2_score":
                    if comparison.r2_delta > 0:
                        return "variant_b"
                    elif comparison.r2_delta < 0:
                        return "variant_a"
            elif primary_metric in ["mae", "mse", "rmse", "latency"]:
                # Lower is better
                if comparison.mae_delta is not None and primary_metric == "mae":
                    if comparison.mae_delta < 0:
                        return "variant_b"
                    elif comparison.mae_delta > 0:
                        return "variant_a"
                elif comparison.latency_delta_ms is not None and primary_metric == "latency":
                    if comparison.latency_delta_ms < 0:
                        return "variant_b"
                    elif comparison.latency_delta_ms > 0:
                        return "variant_a"
            
            return "no_winner"
            
        except Exception as e:
            logger.error(f"Error determining winner: {e}")
            return "inconclusive"
    
    def _generate_recommendation(
        self,
        comparison: ABTestComparison,
        primary_metric: str
    ) -> str:
        """Generate recommendation based on test results"""
        if comparison.winner == "variant_b":
            return "promote_variant_b"
        elif comparison.winner == "variant_a":
            return "keep_variant_a"
        elif comparison.winner == "inconclusive":
            return "continue_testing"
        else:
            return "no_change_needed"
    
    async def _calculate_ab_test_results(self, test_id: str):
        """Calculate final results for a completed A/B test"""
        try:
            async with get_session() as session:
                test = await session.get(ABTestDB, test_id)
                if not test or not test.start_time:
                    return
                
                end_time = test.end_time or datetime.utcnow()
                
                # Calculate comparison
                comparison = await self.calculate_ab_test_metrics(
                    test_id=test_id,
                    start_time=test.start_time,
                    end_time=end_time
                )
                
                # Update test with results
                test.winner = comparison.winner
                test.p_value = comparison.p_value
                test.is_statistically_significant = comparison.is_statistically_significant
                test.confidence_interval_lower = comparison.confidence_interval_lower
                test.confidence_interval_upper = comparison.confidence_interval_upper
                test.variant_a_samples = comparison.variant_a_metrics.total_requests
                test.variant_b_samples = comparison.variant_b_metrics.total_requests
                test.conclusion_reason = f"Winner: {comparison.winner}, Recommendation: {comparison.recommendation}"
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error calculating A/B test results: {e}")
    
    async def store_ab_test_metrics(self, metrics: ABTestMetrics) -> str:
        """Store A/B test metrics in database"""
        try:
            metrics_db = ABTestMetricsDB(
                id=str(uuid.uuid4()),
                test_id=metrics.test_id,
                variant=metrics.variant,
                model_id=metrics.model_id,
                deployment_id=metrics.deployment_id,
                time_window_start=metrics.time_window_start,
                time_window_end=metrics.time_window_end,
                total_requests=metrics.total_requests,
                successful_requests=metrics.successful_requests,
                failed_requests=metrics.failed_requests,
                avg_latency_ms=metrics.avg_latency_ms,
                p50_latency_ms=metrics.p50_latency_ms,
                p95_latency_ms=metrics.p95_latency_ms,
                p99_latency_ms=metrics.p99_latency_ms,
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1_score=metrics.f1_score,
                auc_roc=metrics.auc_roc,
                mae=metrics.mae,
                mse=metrics.mse,
                rmse=metrics.rmse,
                r2_score=metrics.r2_score,
                avg_confidence=metrics.avg_confidence,
                low_confidence_rate=metrics.low_confidence_rate,
                error_rate=metrics.error_rate,
                additional_metrics=metrics.additional_metrics if hasattr(metrics, 'additional_metrics') else None
            )
            
            async with get_session() as session:
                session.add(metrics_db)
                await session.commit()
                return metrics_db.id
                
        except Exception as e:
            logger.error(f"Error storing A/B test metrics: {e}")
            raise

