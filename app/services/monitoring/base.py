"""
Base monitoring service with common utilities and helper methods
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import stats

from app.database import get_session
from sqlalchemy import func, select

logger = logging.getLogger(__name__)


class BaseMonitoringService:
    """Base class for monitoring services with common utilities"""
    
    def __init__(self):
        self.start_time = time.time()
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "error_rate": 5.0,
            "avg_latency_ms": 1000.0
        }
    
    async def _check_database_health(self) -> Tuple[bool, str]:
        """Check the health of the database connection."""
        try:
            async with get_session() as session:
                # Perform a simple query to check connectivity.
                await session.execute(select(func.now()))
            return True, "Database connection successful."
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False, f"Database connection failed: {str(e)}"
    
    def _calculate_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI) between baseline and current distributions"""
        try:
            # Combine both arrays to get consistent bin edges
            combined = np.concatenate([baseline, current])
            min_val, max_val = np.min(combined), np.max(combined)
            
            # Create bins
            if min_val == max_val:
                return 0.0  # No variation, no drift
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate histograms
            baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
            current_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize to probabilities
            baseline_probs = baseline_hist / len(baseline) if len(baseline) > 0 else baseline_hist
            current_probs = current_hist / len(current) if len(current) > 0 else current_hist
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-6
            baseline_probs = baseline_probs + epsilon
            current_probs = current_probs + epsilon
            
            # Calculate PSI
            psi = np.sum((current_probs - baseline_probs) * np.log(current_probs / baseline_probs))
            
            return float(psi)
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    def _ks_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test between baseline and current distributions
        
        Returns:
            Tuple of (KS statistic, p-value)
        """
        try:
            if len(baseline) == 0 or len(current) == 0:
                return 0.0, 1.0
            
            statistic, p_value = stats.ks_2samp(baseline, current)
            return float(statistic), float(p_value)
        except Exception as e:
            logger.error(f"Error performing KS test: {e}")
            return 0.0, 1.0
    
    def _extract_prediction_value(self, output_data: Any) -> Optional[float]:
        """Extract a numeric prediction value from output_data"""
        try:
            if isinstance(output_data, (int, float)):
                return float(output_data)
            elif isinstance(output_data, dict):
                # Try common keys
                for key in ['prediction', 'value', 'score', 'probability', 'output', 'result']:
                    if key in output_data and isinstance(output_data[key], (int, float)):
                        return float(output_data[key])
                # If it's a single value dict, try the first value
                if len(output_data) == 1:
                    val = list(output_data.values())[0]
                    if isinstance(val, (int, float)):
                        return float(val)
            elif isinstance(output_data, list) and len(output_data) > 0:
                # For classification, use max probability or first value
                if isinstance(output_data[0], (int, float)):
                    return float(output_data[0])
        except Exception:
            pass
        return None
    
    def _extract_confidence_score(self, output_data: Any) -> Optional[float]:
        """Extract confidence score from model output data.
        Handles various formats: probabilities, logits, confidence fields, etc.
        """
        try:
            if isinstance(output_data, dict):
                # Try common confidence/probability keys
                confidence_keys = ['confidence', 'probability', 'prob', 'score', 'certainty', 'certainty_score']
                for key in confidence_keys:
                    if key in output_data:
                        val = output_data[key]
                        if isinstance(val, (int, float)):
                            conf = float(val)
                            # Normalize if needed (logits, etc.)
                            if conf < 0 or conf > 1:
                                # Could be logits or other scale, try sigmoid
                                try:
                                    from scipy.special import expit
                                    conf = expit(conf)
                                except:
                                    # If sigmoid fails, clamp to [0, 1]
                                    conf = max(0.0, min(1.0, conf))
                            return conf
                
                # Check for probability array (multi-class)
                if 'probabilities' in output_data or 'probs' in output_data:
                    probs = output_data.get('probabilities') or output_data.get('probs')
                    if isinstance(probs, list) and len(probs) > 0:
                        # Return max probability as confidence
                        return float(max(probs))
                
                # Check for logits
                if 'logits' in output_data:
                    logits = output_data['logits']
                    if isinstance(logits, list) and len(logits) > 0:
                        try:
                            from scipy.special import softmax
                            probs = softmax(logits)
                            return float(max(probs))
                        except:
                            pass
                
            elif isinstance(output_data, list):
                # List of probabilities
                if len(output_data) > 0 and all(isinstance(x, (int, float)) for x in output_data):
                    # Normalize if needed (could be logits)
                    try:
                        from scipy.special import softmax
                        probs = softmax(output_data)
                        return float(max(probs))
                    except:
                        # If not logits, assume probabilities
                        return float(max(output_data))
            
            elif isinstance(output_data, (int, float)):
                # Single value - could be confidence or probability
                conf = float(output_data)
                if conf < 0 or conf > 1:
                    try:
                        from scipy.special import expit
                        conf = expit(conf)
                    except:
                        conf = max(0.0, min(1.0, conf))
                return conf
                
        except Exception as e:
            logger.debug(f"Error extracting confidence: {e}")
            pass
        return None
    
    def _extract_confidence_scores(self, output_data: Any) -> Optional[Dict[str, float]]:
        """Extract full confidence distribution for multi-class predictions"""
        try:
            if isinstance(output_data, dict):
                # Check for class probabilities
                if 'probabilities' in output_data or 'probs' in output_data:
                    probs = output_data.get('probabilities') or output_data.get('probs')
                    if isinstance(probs, (list, dict)):
                        if isinstance(probs, list):
                            return {str(i): float(p) for i, p in enumerate(probs)}
                        else:
                            return {str(k): float(v) for k, v in probs.items()}
                
                # Check for logits
                if 'logits' in output_data:
                    logits = output_data['logits']
                    if isinstance(logits, list):
                        try:
                            from scipy.special import softmax
                            probs = softmax(logits)
                            return {str(i): float(p) for i, p in enumerate(probs)}
                        except:
                            pass
                
                # Check for class-specific confidence
                class_conf = {}
                for key, val in output_data.items():
                    if 'confidence' in key.lower() or 'prob' in key.lower():
                        if isinstance(val, (int, float)):
                            class_conf[key] = float(val)
                if class_conf:
                    return class_conf
                    
        except Exception as e:
            logger.debug(f"Error extracting confidence scores: {e}")
            pass
        return None
    
    def _extract_uncertainty(self, output_data: Any) -> Optional[float]:
        """Extract uncertainty score from model output"""
        try:
            if isinstance(output_data, dict):
                # Try uncertainty-specific keys
                uncertainty_keys = ['uncertainty', 'uncertainty_score', 'entropy', 'variance', 'std', 'std_dev']
                for key in uncertainty_keys:
                    if key in output_data:
                        val = output_data[key]
                        if isinstance(val, (int, float)):
                            return float(val)
                
                # Calculate entropy from probabilities (if available)
                if 'probabilities' in output_data or 'probs' in output_data:
                    probs = output_data.get('probabilities') or output_data.get('probs')
                    if isinstance(probs, list):
                        probs = [max(1e-10, p) for p in probs]  # Avoid log(0)
                        entropy = -sum(p * np.log(p) for p in probs)
                        return float(entropy)
                
                # Calculate variance/std from prediction intervals
                if 'prediction_interval' in output_data or 'interval' in output_data:
                    interval = output_data.get('prediction_interval') or output_data.get('interval')
                    if isinstance(interval, dict) and 'lower' in interval and 'upper' in interval:
                        width = interval['upper'] - interval['lower']
                        # Use width as proxy for uncertainty
                        return float(width)
                        
        except Exception as e:
            logger.debug(f"Error extracting uncertainty: {e}")
            pass
        return None
    
    def _extract_prediction_interval(self, output_data: Any) -> Tuple[Optional[float], Optional[float]]:
        """Extract prediction interval bounds from model output"""
        try:
            if isinstance(output_data, dict):
                # Check for interval fields
                if 'prediction_interval' in output_data:
                    interval = output_data['prediction_interval']
                    if isinstance(interval, dict):
                        lower = interval.get('lower') or interval.get('lower_bound')
                        upper = interval.get('upper') or interval.get('upper_bound')
                        if lower is not None and upper is not None:
                            return float(lower), float(upper)
                
                # Check for separate lower/upper fields
                if 'lower_bound' in output_data and 'upper_bound' in output_data:
                    return float(output_data['lower_bound']), float(output_data['upper_bound'])
                
                if 'interval_lower' in output_data and 'interval_upper' in output_data:
                    return float(output_data['interval_lower']), float(output_data['interval_upper'])
                
        except Exception as e:
            logger.debug(f"Error extracting prediction interval: {e}")
            pass
        return None, None
    
    def _extract_protected_attribute(self, input_data: Dict[str, Any], attribute_name: str) -> Optional[str]:
        """Extract protected attribute value from input data"""
        try:
            if isinstance(input_data, dict):
                return str(input_data.get(attribute_name, ""))
            return None
        except:
            return None
    
    def _extract_ground_truth_value(self, ground_truth: Any) -> Optional[int]:
        """Extract ground truth value as integer (0 or 1 for binary classification)"""
        try:
            if ground_truth is None:
                return None
            
            # Handle various formats
            if isinstance(ground_truth, (int, float)):
                return int(ground_truth)
            elif isinstance(ground_truth, bool):
                return 1 if ground_truth else 0
            elif isinstance(ground_truth, str):
                # Try to parse as number
                try:
                    return int(float(ground_truth))
                except:
                    # Try boolean strings
                    if ground_truth.lower() in ["true", "1", "yes", "positive"]:
                        return 1
                    elif ground_truth.lower() in ["false", "0", "no", "negative"]:
                        return 0
            elif isinstance(ground_truth, dict):
                # Try common keys
                for key in ["label", "value", "target", "ground_truth"]:
                    if key in ground_truth:
                        return self._extract_ground_truth_value(ground_truth[key])
            
            return None
        except:
            return None

