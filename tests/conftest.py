"""
Pytest configuration and fixtures for EasyMLOps tests
Main conftest file that imports all fixtures and sets up test environment
"""

import os
import tempfile
import shutil
from pathlib import Path

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
    
    env_vars_to_preserve = [
        "USE_SQLITE", "SQLITE_PATH", "DATABASE_URL", 
        "DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME",
        "MODELS_DIR", "BENTOS_DIR", "STATIC_DIR"
    ]
    
    for var in env_vars_to_preserve:
        _original_env[var] = os.environ.get(var)
    
    os.environ["USE_SQLITE"] = "true"
    os.environ["SQLITE_PATH"] = str(TEST_DATABASE_PATH)
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["DISABLE_MONITORING"] = "true"
    os.environ["MODELS_DIR"] = str(Path(TEST_DIR) / "models")
    os.environ["BENTOS_DIR"] = str(Path(TEST_DIR) / "bentos")
    os.environ["STATIC_DIR"] = str(Path(TEST_DIR) / "static")
    
    os.makedirs(os.environ["MODELS_DIR"], exist_ok=True)
    os.makedirs(os.environ["BENTOS_DIR"], exist_ok=True)
    os.makedirs(os.environ["STATIC_DIR"], exist_ok=True)


# Set up the test environment
setup_test_environment()

# Now we can safely import the app modules
import pytest
import asyncio
from app.core.app_factory import create_app
from app.database import engine

# Global test app - will be created once
_test_app = None


def get_test_app():
    """Get or create the test app instance"""
    global _test_app
    if _test_app is None:
        from app.config import get_settings
        from app.utils.logging import setup_logging, get_logger
        import app.main
        
        test_settings = get_settings()
        setup_logging()
        test_logger = get_logger(__name__)
        
        app.main.settings = test_settings
        app.main.logger = test_logger
        
        _test_app = create_app(test_settings, test_logger)
    return _test_app


# Import cleanup function from database fixtures
from tests.fixtures.database import cleanup_test_database


def pytest_configure(config):
    """Configure pytest - environment variables already set above"""
    pass


def pytest_unconfigure(config):
    """Clean up after tests"""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_config_for_config_tests(request):
    """Reset environment for config tests that need to test PostgreSQL or default settings"""
    if "test_config.py" in str(request.fspath) and (
        "database_url" in request.node.name.lower() or
        "postgresql" in request.node.name.lower() or
        "complete_configuration" in request.node.name.lower() or
        "file_settings_defaults" in request.node.name.lower() or
        "environment_variable_precedence" in request.node.name.lower()
    ):
        for var, value in _original_env.items():
            if value is not None:
                os.environ[var] = value
            else:
                os.environ.pop(var, None)
        
        yield
        
        setup_test_environment()
    else:
        yield


# Import all fixtures from submodules
from tests.fixtures.database import (
    setup_test_database,
    ensure_clean_database,
    test_session,
    client,
    test_model,
    test_model_validated,
    test_deployment,
    mock_prediction_log,
    mock_performance_metric,
    mock_system_health_metric,
    sample_alerts_list,
    isolated_test_session,
    robust_test_session,
)

from tests.fixtures.services import (
    sample_model_data,
    sample_deployment_data,
    sample_prediction_data,
    monitoring_service,
    deployment_service,
    temp_model_file,
    mock_bentoml_service,
    AsyncMock,
    async_mock,
)

__all__ = [
    # Environment setup
    "TEST_DIR",
    "TEST_DATABASE_PATH",
    "TEST_DATABASE_URL",
    "setup_test_environment",
    "get_test_app",
    "cleanup_test_database",
    # Database fixtures
    "setup_test_database",
    "ensure_clean_database",
    "test_session",
    "client",
    "test_model",
    "test_model_validated",
    "test_deployment",
    "mock_prediction_log",
    "mock_performance_metric",
    "mock_system_health_metric",
    "sample_alerts_list",
    "isolated_test_session",
    "robust_test_session",
    # Service fixtures
    "sample_model_data",
    "sample_deployment_data",
    "sample_prediction_data",
    "monitoring_service",
    "deployment_service",
    "temp_model_file",
    "mock_bentoml_service",
    "AsyncMock",
    "async_mock",
]
