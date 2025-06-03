"""
Pytest configuration and fixtures for EasyMLOps tests
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import get_db, Base
from app.config import get_settings
from app.models.model import Model, Deployment
from app.models.monitoring import (
    PredictionLog, ModelPerformanceMetric, SystemHealthMetric,
    Alert, AuditLog
)


# Test database URL - use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Test settings configuration"""
    settings = get_settings()
    settings.DATABASE_URL = TEST_DATABASE_URL
    settings.DEBUG = True
    settings.MODELS_DIR = tempfile.mkdtemp()
    settings.BENTOS_DIR = tempfile.mkdtemp()
    settings.STATIC_DIR = tempfile.mkdtemp()
    return settings


@pytest.fixture(scope="function")
def test_engine(test_settings):
    """Create test database engine"""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Clean up
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    
    # Remove test database file
    if os.path.exists("test.db"):
        os.remove("test.db")


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create test database session"""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(test_session, test_settings):
    """Create test client with database dependency override"""
    
    def override_get_db():
        try:
            yield test_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_model_data():
    """Sample model data for testing"""
    return {
        "name": "test_model",
        "description": "A test model for unit testing",
        "model_type": "classification",
        "framework": "sklearn",
        "version": "1.0.0",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"}
            },
            "required": ["feature1", "feature2"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "prediction": {"type": "string"},
                "probability": {"type": "number"}
            }
        },
        "performance_metrics": {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97
        }
    }


@pytest.fixture
def sample_deployment_data():
    """Sample deployment data for testing"""
    return {
        "name": "test_deployment",
        "description": "A test deployment",
        "endpoint_url": "http://localhost:3001",
        "environment": "test",
        "resources": {
            "cpu": "100m",
            "memory": "256Mi"
        },
        "scaling": {
            "min_replicas": 1,
            "max_replicas": 3
        }
    }


@pytest.fixture
def sample_prediction_data():
    """Sample prediction request data"""
    return {
        "feature1": 0.5,
        "feature2": "test_value"
    }


@pytest.fixture
def test_model(test_session, sample_model_data):
    """Create a test model in the database"""
    model = Model(**sample_model_data)
    test_session.add(model)
    test_session.commit()
    test_session.refresh(model)
    return model


@pytest.fixture
def test_deployment(test_session, test_model, sample_deployment_data):
    """Create a test deployment in the database"""
    deployment_data = sample_deployment_data.copy()
    deployment_data["model_id"] = test_model.id
    deployment = Deployment(**deployment_data)
    test_session.add(deployment)
    test_session.commit()
    test_session.refresh(deployment)
    return deployment


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing uploads"""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create a simple test model
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    joblib.dump(model, temp_file.name)
    temp_file.close()
    
    yield temp_file.name
    
    # Clean up
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


@pytest.fixture
def mock_prediction_log(test_session, test_model):
    """Create a mock prediction log for testing"""
    log = PredictionLog(
        model_id=test_model.id,
        input_data={"feature1": 0.5, "feature2": "test"},
        prediction={"prediction": "class_a", "probability": 0.85},
        response_time_ms=150.5,
        status="success"
    )
    test_session.add(log)
    test_session.commit()
    test_session.refresh(log)
    return log


@pytest.fixture
def mock_performance_metric(test_session, test_model):
    """Create a mock performance metric for testing"""
    metric = ModelPerformanceMetric(
        model_id=test_model.id,
        metric_name="accuracy",
        metric_value=0.95,
        evaluation_data={"test_samples": 100}
    )
    test_session.add(metric)
    test_session.commit()
    test_session.refresh(metric)
    return metric


@pytest.fixture
def mock_system_health_metric(test_session):
    """Create a mock system health metric for testing"""
    metric = SystemHealthMetric(
        component_name="api_server",
        metric_name="cpu_usage",
        metric_value=45.2,
        unit="percentage",
        threshold_warning=70.0,
        threshold_critical=90.0
    )
    test_session.add(metric)
    test_session.commit()
    test_session.refresh(metric)
    return metric


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_dirs():
    """Clean up test directories after test session"""
    yield
    
    # Clean up any test directories
    for dir_name in ["test_models", "test_bentos", "test_static"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)


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