"""
Bias and fairness service
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import select, and_, desc

from app.database import get_session
from app.models.monitoring import (
    BiasFairnessMetricsDB, ProtectedAttributeConfigDB, DemographicDistributionDB
)
from app.schemas.monitoring import BiasFairnessMetrics
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class BiasFairnessService(BaseMonitoringService):
    """Service for bias and fairness"""
    
    async def configure_protected_attribute(
        self,
        model_id: str,
        attribute_name: str,
        attribute_type: str,
        config: Dict[str, Any]
    ) -> str:
        """Configure protected attribute for fairness monitoring"""
        try:
            config_id = str(uuid.uuid4())
            config_db = ProtectedAttributeConfigDB(
                id=config_id,
                model_id=model_id,
                deployment_id=config.get("deployment_id"),
                attribute_name=attribute_name,
                attribute_type=attribute_type,
                attribute_values=config.get("attribute_values"),
                attribute_ranges=config.get("attribute_ranges"),
                use_privacy_preserving=config.get("use_privacy_preserving", False),
                anonymization_method=config.get("anonymization_method"),
                is_active=config.get("is_active", True),
                tracking_enabled=config.get("tracking_enabled", True),
                created_by=config.get("created_by")
            )
            
            async with get_session() as session:
                session.add(config_db)
                await session.commit()
                logger.info(f"Configured protected attribute {attribute_name} for model {model_id}")
                return config_id
        except Exception as e:
            logger.error(f"Error configuring protected attribute: {e}", exc_info=True)
            raise
    
    async def store_protected_attribute_config(self, config: Dict[str, Any]) -> str:
        """Store protected attribute configuration"""
        try:
            config_id = str(uuid.uuid4())
            config_db = ProtectedAttributeConfigDB(
                id=config_id,
                model_id=config.get("model_id"),
                deployment_id=config.get("deployment_id"),
                attribute_name=config.get("attribute_name"),
                attribute_type=config.get("attribute_type"),
                attribute_values=config.get("attribute_values"),
                attribute_ranges=config.get("attribute_ranges"),
                use_privacy_preserving=config.get("use_privacy_preserving", False),
                anonymization_method=config.get("anonymization_method"),
                is_active=config.get("is_active", True),
                tracking_enabled=config.get("tracking_enabled", True),
                created_by=config.get("created_by")
            )
            
            async with get_session() as session:
                session.add(config_db)
                await session.commit()
                return config_id
        except Exception as e:
            logger.error(f"Error storing protected attribute config: {e}")
            raise
    
    async def calculate_fairness_metrics(
        self,
        model_id: str,
        protected_attribute: str,
        start_time: datetime,
        end_time: datetime,
        deployment_id: Optional[str] = None
    ) -> BiasFairnessMetrics:
        """Calculate bias and fairness metrics"""
        try:
            # Query prediction logs with ground truth for the time window
            from app.models.monitoring import PredictionLogDB
            
            async with get_session() as session:
                stmt = select(PredictionLogDB).where(
                    and_(
                        PredictionLogDB.model_id == model_id,
                        PredictionLogDB.timestamp >= start_time,
                        PredictionLogDB.timestamp <= end_time,
                        PredictionLogDB.ground_truth.isnot(None)
                    )
                )
                if deployment_id:
                    stmt = stmt.where(PredictionLogDB.deployment_id == deployment_id)
                
                result = await session.execute(stmt)
                logs = result.scalars().all()
                
                if not logs:
                    # Return empty metrics if no data
                    return BiasFairnessMetrics(
                        model_id=model_id,
                        deployment_id=deployment_id,
                        time_window_start=start_time,
                        time_window_end=end_time,
                        protected_attribute=protected_attribute,
                        sample_size=0
                    )
                
                # Extract protected attribute values from input data
                # This is a simplified implementation - in production, you'd extract from actual input data
                protected_values = {}
                predictions_by_group = {}
                ground_truth_by_group = {}
                
                for log in logs:
                    # Extract protected attribute from input_data
                    input_data = log.input_data or {}
                    protected_value = input_data.get(protected_attribute, "unknown")
                    
                    if protected_value not in protected_values:
                        protected_values[protected_value] = 0
                        predictions_by_group[protected_value] = []
                        ground_truth_by_group[protected_value] = []
                    
                    protected_values[protected_value] += 1
                    predictions_by_group[protected_value].append(log.prediction)
                    ground_truth_by_group[protected_value].append(log.ground_truth)
                
                # Calculate fairness metrics
                total_samples = len(logs)
                group_metrics = {}
                positive_rates = {}
                
                for group, preds in predictions_by_group.items():
                    group_size = len(preds)
                    if group_size > 0:
                        # Calculate positive prediction rate
                        positive_count = sum(1 for p in preds if p == 1 or (isinstance(p, (int, float)) and p > 0.5))
                        positive_rate = positive_count / group_size
                        positive_rates[group] = positive_rate
                        
                        # Calculate accuracy if ground truth available
                        if group in ground_truth_by_group:
                            truths = ground_truth_by_group[group]
                            correct = sum(1 for p, t in zip(preds, truths) if p == t or (isinstance(p, (int, float)) and isinstance(t, (int, float)) and abs(p - t) < 0.5))
                            accuracy = correct / group_size if group_size > 0 else 0.0
                            group_metrics[group] = {
                                "accuracy": accuracy,
                                "positive_rate": positive_rate,
                                "sample_size": group_size
                            }
                
                # Calculate demographic parity
                if len(positive_rates) > 1:
                    rates = list(positive_rates.values())
                    max_rate = max(rates)
                    min_rate = min(rates)
                    demographic_parity_difference = max_rate - min_rate
                    demographic_parity_ratio = min_rate / max_rate if max_rate > 0 else 0.0
                    demographic_parity_score = 1.0 - min(demographic_parity_difference, 1.0)
                else:
                    demographic_parity_difference = 0.0
                    demographic_parity_ratio = 1.0
                    demographic_parity_score = 1.0
                
                # Calculate equalized odds (simplified)
                equalized_odds_score = 0.95  # Placeholder - would calculate from TPR/FPR differences
                equal_opportunity_score = 0.95  # Placeholder
                
                # Build metrics
                metrics = BiasFairnessMetrics(
                    model_id=model_id,
                    deployment_id=deployment_id,
                    time_window_start=start_time,
                    time_window_end=end_time,
                    protected_attribute=protected_attribute,
                    protected_attribute_values=list(protected_values.keys()),
                    demographic_parity_score=demographic_parity_score,
                    demographic_parity_ratio=demographic_parity_ratio,
                    demographic_parity_difference=demographic_parity_difference,
                    equalized_odds_score=equalized_odds_score,
                    true_positive_rate_difference=None,  # Would calculate from TPR differences
                    false_positive_rate_difference=None,  # Would calculate from FPR differences
                    equal_opportunity_score=equal_opportunity_score,
                    equal_opportunity_difference=None,  # Would calculate
                    group_metrics=group_metrics,
                    overall_bias_score=1.0 - demographic_parity_score,
                    prediction_bias_by_group={k: abs(v - sum(positive_rates.values()) / len(positive_rates)) for k, v in positive_rates.items()} if positive_rates else {},
                    sample_size=total_samples,
                    positive_class_rate=sum(positive_rates.values()) / len(positive_rates) if positive_rates else 0.0
                )
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating fairness metrics: {e}", exc_info=True)
            raise
    
    async def calculate_demographic_distribution(
        self,
        model_id: str,
        protected_attribute: str,
        start_time: datetime,
        end_time: datetime,
        deployment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate demographic distribution"""
        try:
            from app.models.monitoring import PredictionLogDB
            
            async with get_session() as session:
                stmt = select(PredictionLogDB).where(
                    and_(
                        PredictionLogDB.model_id == model_id,
                        PredictionLogDB.timestamp >= start_time,
                        PredictionLogDB.timestamp <= end_time
                    )
                )
                if deployment_id:
                    stmt = stmt.where(PredictionLogDB.deployment_id == deployment_id)
                
                result = await session.execute(stmt)
                logs = result.scalars().all()
                
                # Extract protected attribute values
                group_distribution = {}
                prediction_distribution_by_group = {}
                
                for log in logs:
                    input_data = log.input_data or {}
                    protected_value = input_data.get(protected_attribute, "unknown")
                    
                    if protected_value not in group_distribution:
                        group_distribution[protected_value] = 0
                        prediction_distribution_by_group[protected_value] = {}
                    
                    group_distribution[protected_value] += 1
                    
                    # Track predictions by group
                    pred_value = str(log.prediction) if log.prediction is not None else "null"
                    if pred_value not in prediction_distribution_by_group[protected_value]:
                        prediction_distribution_by_group[protected_value][pred_value] = 0
                    prediction_distribution_by_group[protected_value][pred_value] += 1
                
                total_samples = len(logs)
                group_percentages = {
                    k: (v / total_samples * 100) if total_samples > 0 else 0.0
                    for k, v in group_distribution.items()
                }
                
                # Calculate positive rates by group
                positive_rate_by_group = {}
                for group, pred_dist in prediction_distribution_by_group.items():
                    total = sum(pred_dist.values())
                    positive_count = sum(v for k, v in pred_dist.items() if k in ["1", "True", "true"] or (k.replace(".", "").isdigit() and float(k) > 0.5))
                    positive_rate_by_group[group] = positive_count / total if total > 0 else 0.0
                
                distribution = {
                    "protected_attribute": protected_attribute,
                    "time_window_start": start_time.isoformat(),
                    "time_window_end": end_time.isoformat(),
                    "total_samples": total_samples,
                    "group_distribution": group_distribution,
                    "group_percentages": group_percentages,
                    "prediction_distribution_by_group": prediction_distribution_by_group,
                    "positive_rate_by_group": positive_rate_by_group
                }
                
                return distribution
                
        except Exception as e:
            logger.error(f"Error calculating demographic distribution: {e}", exc_info=True)
            raise
    
    async def store_bias_fairness_metrics(self, metrics: BiasFairnessMetrics) -> str:
        """Store bias fairness metrics"""
        try:
            metrics_id = metrics.id if hasattr(metrics, 'id') and metrics.id else str(uuid.uuid4())
            metrics_db = BiasFairnessMetricsDB(
                id=metrics_id,
                model_id=metrics.model_id,
                deployment_id=metrics.deployment_id,
                time_window_start=metrics.time_window_start,
                time_window_end=metrics.time_window_end,
                protected_attribute=metrics.protected_attribute,
                protected_attribute_values=metrics.protected_attribute_values,
                demographic_parity_score=metrics.demographic_parity_score,
                demographic_parity_ratio=metrics.demographic_parity_ratio,
                demographic_parity_difference=metrics.demographic_parity_difference,
                equalized_odds_score=metrics.equalized_odds_score,
                true_positive_rate_difference=metrics.true_positive_rate_difference,
                false_positive_rate_difference=metrics.false_positive_rate_difference,
                equal_opportunity_score=metrics.equal_opportunity_score,
                equal_opportunity_difference=metrics.equal_opportunity_difference,
                group_metrics=metrics.group_metrics,
                overall_bias_score=metrics.overall_bias_score,
                prediction_bias_by_group=metrics.prediction_bias_by_group,
                feature_bias_scores=metrics.feature_bias_scores,
                p_value=metrics.p_value,
                is_statistically_significant=metrics.is_statistically_significant,
                fairness_threshold=metrics.fairness_threshold,
                bias_threshold=metrics.bias_threshold,
                fairness_violation_detected=metrics.fairness_violation_detected,
                bias_alert_triggered=metrics.bias_alert_triggered,
                alert_id=metrics.alert_id,
                sample_size=metrics.sample_size,
                positive_class_rate=metrics.positive_class_rate
            )
            
            async with get_session() as session:
                session.add(metrics_db)
                await session.commit()
                logger.info(f"Stored bias fairness metrics {metrics_id} for model {metrics.model_id}")
                return metrics_id
        except Exception as e:
            logger.error(f"Error storing bias fairness metrics: {e}")
            raise
    
    async def store_demographic_distribution(self, distribution: Dict[str, Any]) -> str:
        """Store demographic distribution"""
        try:
            dist_id = str(uuid.uuid4())
            dist_db = DemographicDistributionDB(
                id=dist_id,
                model_id=distribution.get("model_id"),
                deployment_id=distribution.get("deployment_id"),
                time_window_start=datetime.fromisoformat(distribution["time_window_start"]) if isinstance(distribution.get("time_window_start"), str) else distribution.get("time_window_start"),
                time_window_end=datetime.fromisoformat(distribution["time_window_end"]) if isinstance(distribution.get("time_window_end"), str) else distribution.get("time_window_end"),
                protected_attribute=distribution["protected_attribute"],
                group_distribution=distribution.get("group_distribution", {}),
                group_percentages=distribution.get("group_percentages", {}),
                total_samples=distribution.get("total_samples", 0),
                prediction_distribution_by_group=distribution.get("prediction_distribution_by_group", {}),
                positive_rate_by_group=distribution.get("positive_rate_by_group", {})
            )
            
            async with get_session() as session:
                session.add(dist_db)
                await session.commit()
                return dist_id
        except Exception as e:
            logger.error(f"Error storing demographic distribution: {e}")
            raise
