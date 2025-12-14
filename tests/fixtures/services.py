"""
Service-related fixtures for testing
Includes service instances, mock fixtures, and test data fixtures
"""

import os
import tempfile
import uuid
import pytest

from app.services.monitoring_service import MonitoringService
from app.services.deployment_service import DeploymentService


@pytest.fixture
def sample_model_data():
    """Sample model data for testing"""
    return {
        "name": "test_model",
        "description": "A test model for unit testing",
        "model_type": "classification",
        "framework": "sklearn",
        "version": "1.0.0",
        "file_name": "test_model.joblib",
        "file_size": 1024,
        "file_hash": f"sample_hash_{uuid.uuid4().hex[:8]}"
    }


@pytest.fixture
def sample_deployment_data():
    """Sample deployment data for testing"""
    return {
        "deployment_name": "test_deployment",
        "deployment_url": "http://localhost:3001",
        "status": "active",
        "configuration": {
            "cpu": "100m",
            "memory": "256Mi"
        },
        "cpu_request": 0.1,
        "memory_request": "256Mi",
        "replicas": 1,
        "framework": "sklearn",
        "endpoints": ["predict", "predict_proba"]
    }


@pytest.fixture
def sample_prediction_data():
    """Sample prediction request data"""
    return {
        "feature1": 0.5,
        "feature2": "test_value"
    }


@pytest.fixture
def monitoring_service():
    """Create monitoring service instance"""
    return MonitoringService()


@pytest.fixture
def deployment_service():
    """Create deployment service instance"""
    return DeploymentService()


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing uploads"""
    # Optional imports
    try:
        import joblib
        JOBLIB_AVAILABLE = True
    except ImportError:
        joblib = None
        JOBLIB_AVAILABLE = False
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False
        pytest.skip("scikit-learn not available - cannot create test model")
    
    # Create a simple test model
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    
    if JOBLIB_AVAILABLE:
        joblib.dump(model, temp_file.name)
    else:
        # Fallback to pickle
        import pickle
        with open(temp_file.name, 'wb') as f:
            pickle.dump(model, f)
    
    temp_file.close()
    
    yield temp_file.name
    
    # Clean up
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


@pytest.fixture
def mock_bentoml_service():
    """Mock BentoML service for testing"""
    class MockBentoService:
        def __init__(self, name="test_service"):
            self.name = name
            self.version = "1.0.0"
            self.path = f"/tmp/{name}"
        
        def predict(self, input_data):
            return {"prediction": "test_result", "probability": 0.95}
    
    return MockBentoService()


class AsyncMock:
    """Simple async mock for testing async functions"""
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_count = 0
        self.call_args_list = []
    
    async def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.call_args_list.append((args, kwargs))
        
        if self.side_effect:
            if isinstance(self.side_effect, Exception):
                raise self.side_effect
            elif callable(self.side_effect):
                return await self.side_effect(*args, **kwargs)
        
        return self.return_value


@pytest.fixture
def async_mock():
    """Factory for creating async mocks"""
    return AsyncMock

