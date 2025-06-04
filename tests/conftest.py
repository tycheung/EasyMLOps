"""
Pytest configuration and fixtures for EasyMLOps tests
"""

import os
import tempfile
import shutil
from pathlib import Path
import uuid
from datetime import datetime, timedelta, timezone

# IMPORTANT: Set environment variables BEFORE any app imports
# This ensures the database configuration is correct from the start
TEST_DIR = tempfile.mkdtemp(prefix="easymlops_test_")
TEST_DATABASE_PATH = Path(TEST_DIR) / "test.db"
TEST_DATABASE_URL = f"sqlite:///{TEST_DATABASE_PATH}"

# Store original environment for config tests
_original_env = {}

def setup_test_environment():
    """Set up test environment, preserving original for config tests"""
    global _original_env
    
    # Store original environment variables
    env_vars_to_preserve = [
        "USE_SQLITE", "SQLITE_PATH", "DATABASE_URL", 
        "DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME",
        "MODELS_DIR", "BENTOS_DIR", "STATIC_DIR"
    ]
    
    for var in env_vars_to_preserve:
        _original_env[var] = os.environ.get(var)
    
    # Set SQLite configuration for most tests
    os.environ["USE_SQLITE"] = "true"
    os.environ["SQLITE_PATH"] = str(TEST_DATABASE_PATH)
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    
    # Disable monitoring service during tests to prevent database conflicts
    os.environ["DISABLE_MONITORING"] = "true"
    
    # Set other test environment variables
    os.environ["MODELS_DIR"] = str(Path(TEST_DIR) / "models")
    os.environ["BENTOS_DIR"] = str(Path(TEST_DIR) / "bentos")
    os.environ["STATIC_DIR"] = str(Path(TEST_DIR) / "static")
    
    # Create test directories
    os.makedirs(os.environ["MODELS_DIR"], exist_ok=True)
    os.makedirs(os.environ["BENTOS_DIR"], exist_ok=True)
    os.makedirs(os.environ["STATIC_DIR"], exist_ok=True)

# Set up the test environment
setup_test_environment()

# Now we can safely import the app modules
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from sqlmodel import Session
from sqlalchemy.ext.asyncio import AsyncSession

# Import app creation function instead of the app itself
from app.main import create_app
from app.database import get_db, engine
from app.config import get_settings
from app.models.model import Model, ModelDeployment
from app.models.monitoring import (
    PredictionLogDB, ModelPerformanceMetricsDB, SystemHealthMetricDB,
    AlertDB, AuditLogDB
)
from app.schemas.monitoring import (
    AlertSeverity, SystemComponent, Alert
)


# Global test app - will be created once
_test_app = None


def get_test_app():
    """Get or create the test app instance"""
    global _test_app
    if _test_app is None:
        # Initialize settings and logger for app creation
        from app.config import get_settings
        from app.utils.logging import setup_logging, get_logger
        import app.main
        
        # Get settings and logger instances
        test_settings = get_settings()
        setup_logging()
        test_logger = get_logger(__name__)
        
        # Set global variables for compatibility
        app.main.settings = test_settings
        app.main.logger = test_logger
        
        # Create app with explicit settings and logger
        _test_app = create_app(test_settings, test_logger)
    return _test_app


def pytest_configure(config):
    """Configure pytest - environment variables already set above"""
    pass  # Environment variables are already set at module level


def pytest_unconfigure(config):
    """Clean up after tests"""
    # Remove temporary test directory
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Set up the test database once for the entire test session"""
    # Import here to ensure environment variables are set
    from app.database import create_tables
    from sqlmodel import SQLModel
    from app.database import Base
    
    # Create all tables
    SQLModel.metadata.create_all(engine)
    Base.metadata.create_all(engine)
    
    yield
    
    # Cleanup happens in pytest_unconfigure


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def cleanup_test_database():
    """Utility function to clean up all test data from the database"""
    max_retries = 3
    retry_delay = 0.1  # Start with 100ms delay
    
    for attempt in range(max_retries):
        try:
            with Session(engine) as session:
                # Import all model classes that might have data
                from app.models.model import Model, ModelDeployment
                from app.models.monitoring import (
                    PredictionLogDB, ModelPerformanceMetricsDB, SystemHealthMetricDB,
                    AlertDB, AuditLogDB
                )
                
                # Delete in reverse dependency order to avoid foreign key issues
                session.query(AlertDB).delete()
                session.query(AuditLogDB).delete()
                session.query(SystemHealthMetricDB).delete()
                session.query(ModelPerformanceMetricsDB).delete()
                session.query(PredictionLogDB).delete()
                session.query(ModelDeployment).delete()
                session.query(Model).delete()
                
                # Commit the cleanup
                session.commit()
                return  # Success, exit retry loop
                
        except Exception as e:
            if "closed database" not in str(e).lower():
                if attempt < max_retries - 1:
                    import time
                    print(f"Warning: Database cleanup attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Warning: Database cleanup failed after {max_retries} attempts: {e}")


@pytest.fixture(autouse=True)
def ensure_clean_database():
    """Ensure database is clean before each test starts"""
    # Clean up before test
    cleanup_test_database()
    yield
    # Cleanup after test is handled by test_session fixture


@pytest.fixture
def test_session():
    """Create a test database session for each test with proper cleanup and connection management"""
    session = None
    try:
        session = Session(engine)
        yield session
        
    finally:
        # Clean up after test - ensure test isolation
        if session:
            try:
                # First rollback any uncommitted changes
                session.rollback()
                session.close()  # Explicitly close the session
                
            except Exception as e:
                if "closed database" not in str(e).lower():
                    print(f"Warning: Session cleanup failed: {e}")
        
        # Then clean up all test data
        cleanup_test_database()


@pytest.fixture
def client(test_session):
    """Create test client with database dependency override"""
    global _test_app # Ensure we are modifying the global _test_app
    _test_app = None # Force app recreation for this client fixture

    # engine is the synchronous test engine defined in this file
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async_session_for_test = AsyncSession(engine) # Create AsyncSession with the sync test engine
        try:
            yield async_session_for_test
            # The app's get_db will handle commit/rollback based on its logic
            # We might still want to ensure a final commit or rollback here if the app doesn't always.
            # For now, let app's get_db manage it.
        except Exception:
            # await async_session_for_test.rollback() # App's get_db should handle this
            raise
        finally:
            await async_session_for_test.close()
            # The original synchronous test_session might be used for direct setup in some tests.
            # Its cleanup is handled by its own fixture or ensure_clean_database.
            # cleanup_test_database() is called by ensure_clean_database fixture (autouse=True)
            # and also within the test_session fixture's finally block.
            # To avoid duplicate cleanup or potential issues, let's ensure it's called reliably once after.
            # The ensure_clean_database fixture should handle the main cleanup.

    get_test_app().dependency_overrides[get_db] = override_get_db
    
    with TestClient(get_test_app()) as test_client:
        yield test_client
    
    get_test_app().dependency_overrides.clear()


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
        "file_hash": f"sample_hash_{uuid.uuid4().hex[:8]}"  # Unique hash for each usage
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
def test_model(test_session):
    """Create a test model instance for testing"""
    model = Model(
        name="test_model",
        description="A test model for unit testing",
        model_type="classification",
        framework="sklearn",
        version="1.0.0",
        file_name="test_model.joblib",
        file_size=1024,
        file_hash=f"test_hash_{uuid.uuid4().hex[:8]}",  # Unique hash for each test
        status="uploaded"
    )
    
    test_session.add(model)
    test_session.commit()
    test_session.refresh(model)
    
    return model


@pytest.fixture
def test_deployment(test_session, test_model, sample_deployment_data):
    """Create a test deployment in the database"""
    deployment_data = sample_deployment_data.copy()
    deployment_data["model_id"] = test_model.id
    deployment = ModelDeployment(**deployment_data)
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
    log = PredictionLogDB(
        id="test_log_123",
        model_id=test_model.id,
        request_id="req_123",
        input_data={"feature1": 0.5, "feature2": "test"},
        output_data={"prediction": "class_a", "probability": 0.85},
        latency_ms=150.5,
        api_endpoint="/predict",
        success=True
    )
    test_session.add(log)
    test_session.commit()
    test_session.refresh(log)
    return log


@pytest.fixture
def mock_performance_metric(test_session, test_model):
    """Create a mock performance metric for testing"""
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    
    metric = ModelPerformanceMetricsDB(
        id="test_metric_123",
        model_id=test_model.id,
        time_window_start=now - timedelta(hours=1),
        time_window_end=now,
        total_requests=100,
        successful_requests=95,
        failed_requests=5,
        requests_per_minute=10.0,
        avg_latency_ms=150.0,
        p50_latency_ms=140.0,
        p95_latency_ms=200.0,
        p99_latency_ms=250.0,
        max_latency_ms=300.0,
        success_rate=0.95,
        error_rate=0.05
    )
    test_session.add(metric)
    test_session.commit()
    test_session.refresh(metric)
    return metric


@pytest.fixture
def mock_system_health_metric(test_session):
    """Create a mock system health metric for testing"""
    metric = SystemHealthMetricDB(
        id="test_health_123",
        component="api_server",
        metric_type="cpu_usage",
        value=45.2,
        unit="percentage"
    )
    test_session.add(metric)
    test_session.commit()
    test_session.refresh(metric)
    return metric


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


@pytest.fixture(autouse=True)
def reset_config_for_config_tests(request):
    """Reset environment for config tests that need to test PostgreSQL or default settings"""
    # Check if this is a config test that needs original environment
    if "test_config.py" in str(request.fspath) and (
        "database_url" in request.node.name.lower() or
        "postgresql" in request.node.name.lower() or
        "complete_configuration" in request.node.name.lower() or
        "file_settings_defaults" in request.node.name.lower() or
        "environment_variable_precedence" in request.node.name.lower()
    ):
        # Temporarily restore original environment for these specific tests
        for var, value in _original_env.items():
            if value is not None:
                os.environ[var] = value
            else:
                os.environ.pop(var, None)
        
        yield
        
        # Restore test environment after the config test
        setup_test_environment()
    else:
        # Regular test - keep SQLite configuration
        yield


@pytest.fixture
def isolated_test_session():
    """
    Create an isolated test database session that uses transactions for better isolation.
    This is an alternative to test_session for tests that need stronger isolation guarantees.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()  # Always rollback the transaction
        connection.close() 


@pytest.fixture
def robust_test_session():
    """
    Create a more robust test database session with better error handling
    for tests that experience database locking issues
    """
    max_retries = 3
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        session = None
        try:
            session = Session(engine)
            yield session
            return  # Success, exit retry loop
            
        except Exception as e:
            if session:
                try:
                    session.rollback()
                    session.close()
                except:
                    pass
            
            if attempt < max_retries - 1:
                import time
                print(f"Warning: Session creation attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise  # Re-raise the exception after all retries
        
        finally:
            if session:
                try:
                    session.close()
                except:
                    pass
            
            # Clean up test data
            cleanup_test_database() 


@pytest.fixture
def sample_alerts_list(test_session):
    """Create sample alerts for testing, one active, one inactive"""
    test_session.query(AlertDB).delete() # Clear existing alerts
    
    alert1_data = {
        "id": "alert_1",
        "severity": AlertSeverity.WARNING,
        "component": SystemComponent.API_SERVER,
        "title": "High CPU Usage",
        "description": "CPU usage is high",
        "triggered_at": datetime.now(timezone.utc) - timedelta(hours=1),
        "is_active": True, # Active
        "is_acknowledged": False
    }
    alert2_data = {
        "id": "alert_2",
        "severity": AlertSeverity.CRITICAL,
        "component": SystemComponent.MODEL_SERVICE,
        "title": "Model Degradation",
        "description": "Model accuracy dropped significantly",
        "triggered_at": datetime.now(timezone.utc) - timedelta(minutes=30),
        "is_active": True, # <<<< Make this active too
        "is_acknowledged": False,
        "affected_models": ["model_abc", "model_xyz"]
    }

    alert1_db = AlertDB(**alert1_data)
    alert2_db = AlertDB(**alert2_data)
    test_session.add_all([alert1_db, alert2_db])
    test_session.commit()

    # Convert to Alert schema instances manually
    return [
        Alert(
            id=alert1_db.id,
            severity=alert1_db.severity,
            component=alert1_db.component,
            title=alert1_db.title,
            description=alert1_db.description,
            triggered_at=alert1_db.triggered_at,
            resolved_at=alert1_db.resolved_at,
            acknowledged_at=alert1_db.acknowledged_at,
            acknowledged_by=alert1_db.acknowledged_by,
            metric_value=alert1_db.metric_value,
            threshold_value=alert1_db.threshold_value,
            affected_models=alert1_db.affected_models or [],
            is_active=alert1_db.is_active,
            is_acknowledged=alert1_db.is_acknowledged
        ),
        Alert(
            id=alert2_db.id,
            severity=alert2_db.severity,
            component=alert2_db.component,
            title=alert2_db.title,
            description=alert2_db.description,
            triggered_at=alert2_db.triggered_at,
            resolved_at=alert2_db.resolved_at,
            acknowledged_at=alert2_db.acknowledged_at,
            acknowledged_by=alert2_db.acknowledged_by,
            metric_value=alert2_db.metric_value,
            threshold_value=alert2_db.threshold_value,
            affected_models=alert2_db.affected_models or [],
            is_active=alert2_db.is_active,
            is_acknowledged=alert2_db.is_acknowledged
        )
    ] 