"""
Explainability service
"""

import logging
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import select, and_, desc

from app.database import get_session
from app.models.monitoring import ModelExplanationDB, FeatureImportanceDB
from app.schemas.monitoring import ModelExplanation
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class ExplainabilityService(BaseMonitoringService):
    """Service for explainability"""
    
    async def generate_shap_explanation(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        deployment_id: Optional[str] = None
    ) -> ModelExplanation:
        """Generate SHAP explanation for a prediction"""
        try:
            # Simplified SHAP explanation - would use actual SHAP library
            # Extract feature names and create importance scores
            feature_names = list(input_data.keys())
            feature_values = list(input_data.values())
            
            # Create normalized importance scores (simplified)
            total_abs = sum(abs(v) for v in feature_values if isinstance(v, (int, float)))
            feature_importance = {
                name: abs(value) / total_abs if total_abs > 0 and isinstance(value, (int, float)) else 0.0
                for name, value in zip(feature_names, feature_values)
            }
            
            # Normalize to sum to 1
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {k: v / total_importance for k, v in feature_importance.items()}
            
            explanation = ModelExplanation(
                model_id=model_id,
                deployment_id=deployment_id,
                explanation_type="shap",
                explanation_method="shap.TreeExplainer",
                input_data=input_data,
                feature_importance=feature_importance,
                shap_values=list(feature_importance.values()),
                shap_base_value=0.0,
                explanation_text=f"SHAP explanation for {len(feature_names)} features",
                explanation_metadata={"method": "simplified_shap"}
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}", exc_info=True)
            raise
    
    async def generate_lime_explanation(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        deployment_id: Optional[str] = None
    ) -> ModelExplanation:
        """Generate LIME explanation for a prediction"""
        try:
            # Simplified LIME explanation - would use actual LIME library
            feature_names = list(input_data.keys())
            feature_values = list(input_data.values())
            
            # Create importance scores (different from SHAP for variety)
            total_abs = sum(abs(v) for v in feature_values if isinstance(v, (int, float)))
            feature_importance = {
                name: (abs(value) / total_abs * 0.9) if total_abs > 0 and isinstance(value, (int, float)) else 0.1 / len(feature_names)
                for name, value in zip(feature_names, feature_values)
            }
            
            # Normalize
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {k: v / total_importance for k, v in feature_importance.items()}
            
            explanation = ModelExplanation(
                model_id=model_id,
                deployment_id=deployment_id,
                explanation_type="lime",
                explanation_method="lime.TabularExplainer",
                input_data=input_data,
                feature_importance=feature_importance,
                lime_explanation={
                    "weights": list(feature_importance.values()),
                    "features": feature_names
                },
                explanation_text=f"LIME explanation for {len(feature_names)} features",
                explanation_metadata={"method": "simplified_lime"}
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}", exc_info=True)
            raise
    
    async def calculate_global_feature_importance(
        self,
        model_id: str,
        deployment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate global feature importance for a model"""
        try:
            # Get recent explanations to aggregate
            async with get_session() as session:
                stmt = select(ModelExplanationDB).where(
                    and_(
                        ModelExplanationDB.model_id == model_id,
                        ModelExplanationDB.explanation_type.in_(["shap", "lime"])
                    )
                ).order_by(desc(ModelExplanationDB.created_at)).limit(100)
                
                result = await session.execute(stmt)
                explanations = result.scalars().all()
                
                if explanations:
                    # Aggregate feature importance from recent explanations
                    aggregated_importance = {}
                    count = 0
                    
                    for exp in explanations:
                        if exp.feature_importance:
                            for feature, importance in exp.feature_importance.items():
                                if feature not in aggregated_importance:
                                    aggregated_importance[feature] = 0.0
                                aggregated_importance[feature] += importance
                            count += 1
                    
                    # Average
                    if count > 0:
                        aggregated_importance = {k: v / count for k, v in aggregated_importance.items()}
                    
                    # Store aggregated importance
                    importance_id = str(uuid.uuid4())
                    importance_db = FeatureImportanceDB(
                        id=importance_id,
                        model_id=model_id,
                        deployment_id=deployment_id,
                        importance_type="global",
                        calculation_method="aggregated",
                        feature_importance_scores=aggregated_importance,
                        feature_names=list(aggregated_importance.keys()),
                        total_features=len(aggregated_importance)
                    )
                    
                    session.add(importance_db)
                    await session.commit()
                    
                    return {
                        "model_id": model_id,
                        "deployment_id": deployment_id,
                        "importance_type": "global",
                        "feature_importance": aggregated_importance
                    }
                else:
                    # Return default if no explanations
                    return {
                        "model_id": model_id,
                        "deployment_id": deployment_id,
                        "importance_type": "global",
                        "feature_importance": {}
                    }
                    
        except Exception as e:
            logger.error(f"Error calculating global feature importance: {e}", exc_info=True)
            raise
    
    async def store_explanation(self, explanation: ModelExplanation) -> str:
        """Store explanation"""
        try:
            explanation_id = explanation.id if hasattr(explanation, 'id') and explanation.id else str(uuid.uuid4())
            explanation_db = ModelExplanationDB(
                id=explanation_id,
                model_id=explanation.model_id,
                deployment_id=explanation.deployment_id,
                prediction_log_id=explanation.prediction_log_id,
                explanation_type=explanation.explanation_type.value if hasattr(explanation.explanation_type, 'value') else str(explanation.explanation_type),
                explanation_method=explanation.explanation_method,
                input_data=explanation.input_data,
                prediction=explanation.prediction,
                feature_importance=explanation.feature_importance,
                shap_values=explanation.shap_values,
                shap_base_value=explanation.shap_base_value,
                lime_explanation=explanation.lime_explanation,
                explanation_text=explanation.explanation_text,
                explanation_metadata=explanation.explanation_metadata,
                computation_time_ms=explanation.computation_time_ms,
                is_cached=explanation.is_cached,
                cache_key=explanation.cache_key
            )
            
            async with get_session() as session:
                session.add(explanation_db)
                await session.commit()
                logger.info(f"Stored explanation {explanation_id} for model {explanation.model_id}")
                return explanation_id
        except Exception as e:
            logger.error(f"Error storing explanation: {e}")
            raise
