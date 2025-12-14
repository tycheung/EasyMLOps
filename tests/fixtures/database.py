"""
Database-related fixtures for testing
Includes session fixtures, model fixtures, and database setup/cleanup utilities
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
import pytest
from sqlmodel import Session
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.testclient import TestClient

from app.database import engine, get_db
from app.models.model import Model, ModelDeployment
from app.models.monitoring import (
    PredictionLogDB, ModelPerformanceMetricsDB, SystemHealthMetricDB,
    AlertDB, AuditLogDB
)
from app.schemas.monitoring import AlertSeverity, SystemComponent, Alert
from app.database import engine, get_db


def cleanup_test_database():
    """Utility function to clean up all test data from the database"""
    max_retries = 3
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            with Session(engine) as session:
                session.query(AlertDB).delete()
                session.query(AuditLogDB).delete()
                session.query(SystemHealthMetricDB).delete()
                session.query(ModelPerformanceMetricsDB).delete()
                session.query(PredictionLogDB).delete()
                session.query(ModelDeployment).delete()
                session.query(Model).delete()
                session.commit()
                return
                
        except Exception as e:
            if "closed database" not in str(e).lower():
                if attempt < max_retries - 1:
                    import time
                    print(f"Warning: Database cleanup attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Warning: Database cleanup failed after {max_retries} attempts: {e}")


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Set up the test database once for the entire test session"""
    from app.database import create_tables
    from sqlmodel import SQLModel
    from app.database import Base
    
    SQLModel.metadata.create_all(engine)
    Base.metadata.create_all(engine)
    
    yield
    
    # Cleanup happens in pytest_unconfigure


@pytest.fixture(autouse=True)
def ensure_clean_database():
    """Ensure database is clean before each test starts"""
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
        if session:
            try:
                session.rollback()
                session.close()
            except Exception as e:
                if "closed database" not in str(e).lower():
                    print(f"Warning: Session cleanup failed: {e}")
        
        cleanup_test_database()


@pytest.fixture
def client(test_session):
    """Create test client with database dependency override"""
    # Import here to avoid circular dependency
    from tests.conftest import get_test_app
    import tests.conftest as conftest_module
    conftest_module._test_app = None  # Force app recreation for this client fixture

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async_session_for_test = AsyncSession(engine)
        try:
            yield async_session_for_test
        except Exception:
            raise
        finally:
            await async_session_for_test.close()

    test_app = get_test_app()
    test_app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(test_app) as test_client:
        yield test_client
    
    test_app.dependency_overrides.clear()


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
        file_hash=f"test_hash_{uuid.uuid4().hex[:8]}",
        status="uploaded"
    )
    
    test_session.add(model)
    test_session.commit()
    test_session.refresh(model)
    
    return model


@pytest.fixture
def test_model_validated(test_session):
    """Create a validated test model instance for testing"""
    model = Model(
        name="test_model_validated",
        description="A validated test model for unit testing",
        model_type="classification",
        framework="sklearn",
        version="1.0.0",
        file_name="test_model.joblib",
        file_size=1024,
        file_hash=f"test_hash_validated_{uuid.uuid4().hex[:8]}",
        status="validated"
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
        min_latency_ms=100.0,
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
def sample_alerts_list(test_session):
    """Create sample alerts for testing, one active, one inactive"""
    test_session.query(AlertDB).delete()
    
    alert1_data = {
        "id": "alert_1",
        "severity": AlertSeverity.WARNING,
        "component": SystemComponent.API_SERVER,
        "title": "High CPU Usage",
        "description": "CPU usage is high",
        "triggered_at": datetime.now(timezone.utc) - timedelta(hours=1),
        "is_active": True,
        "is_acknowledged": False
    }
    alert2_data = {
        "id": "alert_2",
        "severity": AlertSeverity.CRITICAL,
        "component": SystemComponent.MODEL_SERVICE,
        "title": "Model Degradation",
        "description": "Model accuracy dropped significantly",
        "triggered_at": datetime.now(timezone.utc) - timedelta(minutes=30),
        "is_active": True,
        "is_acknowledged": False,
        "affected_models": ["model_abc", "model_xyz"]
    }

    alert1_db = AlertDB(**alert1_data)
    alert2_db = AlertDB(**alert2_data)
    test_session.add_all([alert1_db, alert2_db])
    test_session.commit()

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
        transaction.rollback()
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
            return
            
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
                raise
        
        finally:
            if session:
                try:
                    session.close()
                except:
                    pass
            
            cleanup_test_database()

